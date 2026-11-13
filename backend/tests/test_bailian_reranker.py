import json
import os
import time
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

from app.config import BailianRerankSettings, get_llm_settings, get_agent_limits
from app.services import bailian_reranker as b
from app.services.llm_client import ProviderError
from app.services.retrieval_enhancement import candidate_payload, rerank_candidates
from app.services.smoke_diagnostics import sanitize_retrieval_enhancement
from tests import test_retrieval_enhancement as fixtures


class BailianAdapterTests(unittest.TestCase):
    def test_valid_mapping_uses_index_and_score(self):
        rows=b.validate_results({'output':{'results':[{'index':0,'relevance_score':.2},
            {'index':1,'relevance_score':.9}]}},2)
        self.assertEqual(rows,[(1,.9),(0,.2)])

    def test_invalid_indices_duplicates_missing_scores(self):
        for rows in [[],[{'index':0,'relevance_score':.5}],
                     [{'index':0,'relevance_score':.5}]*2,
                     [{'index':True,'relevance_score':.5},{'index':1,'relevance_score':.3}],
                     [{'index':-1,'relevance_score':.5},{'index':1,'relevance_score':.3}],
                     [{'index':2,'relevance_score':.5},{'index':1,'relevance_score':.3}]]:
            with self.subTest(rows=rows),self.assertRaises(ProviderError):b.validate_results({'output':{'results':rows}},2)
        for score in [None,True,'0.5',float('nan'),float('inf'),-.1,1.1]:
            with self.subTest(score=score),self.assertRaises(ProviderError):
                b.validate_results({'output':{'results':[{'index':0,'relevance_score':score}]}},1)
        for value in [None,[],{}, {'output':[]},{'code':'BadRequest'}]:
            with self.assertRaises(ProviderError):b.validate_results(value,2)

    def test_request_credentials_input_limit_and_no_retries(self):
        def transport(request,**kw):
            self.assertEqual(request.get_header('Authorization'),'Bearer bailian-only')
            self.assertNotIn('deepseek-key',str(request.header_items()))
            self.assertEqual(json.loads(request.data)['parameters']['top_n'],2)
            kw['audit'].update(network_state='started')
            return [{'http_status':200},{'data':{'output':{'results':[
                {'index':1,'relevance_score':.9},{'index':0,'relevance_score':.1}]},'usage':{'total_tokens':20}}}]
        audit={}
        with patch.object(b,'isolated_response',side_effect=transport) as mocked:
            self.assertEqual(b.rerank('q',['a','b'],deadline=time.monotonic()+3,audit=audit,
                settings=BailianRerankSettings(api_key='bailian-only'))[0],(1,.9))
            self.assertEqual(mocked.call_count,1)
        self.assertEqual(audit['usage']['total_tokens'],20)
        self.assertEqual(audit['http_attempts'],1)
        with patch.object(b,'isolated_response') as mocked:
            for query,docs in [('x'*4001,['a']),('q',['a'*4001]),('q',['a']*501)]:
                with self.assertRaises(ProviderError):b.rerank(query,docs,deadline=time.monotonic()+3,audit={},settings=BailianRerankSettings(api_key='x'))
            mocked.assert_not_called()

    def test_timeout_auth_server_and_network_errors(self):
        for result,code in [([{'error':'http','status':401}],'rerank_auth_failed'),
                            ([{'error':'http','status':403}],'rerank_auth_failed'),
                            ([{'error':'http','status':503}],'rerank_http_503'),
                            ([{'error':'timeout'}],'rerank_timeout'),
                            ([{'error':'network'}],'rerank_network')]:
            with patch.object(b,'isolated_response',return_value=result) as mocked:
                with self.assertRaises(ProviderError) as e:
                    b.rerank('q',['a','b'],deadline=time.monotonic()+2,audit={},settings=BailianRerankSettings(api_key='x'))
                self.assertEqual(e.exception.code,code);self.assertEqual(mocked.call_count,1)
        with patch.object(b,'isolated_response',side_effect=TimeoutError):
            with self.assertRaises(ProviderError) as e:b.rerank('q',['a'],deadline=time.monotonic()+1,audit={},settings=BailianRerankSettings(api_key='x'))
            self.assertEqual(e.exception.code,'rerank_timeout')

    def test_official_beijing_endpoint_only(self):
        self.assertTrue(b.validate_endpoint(BailianRerankSettings.endpoint))
        self.assertTrue(b.validate_endpoint('https://workspace-123.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'))
        for url in ['http://dashscope.aliyuncs.com','https://evil.invalid/api/v1/services/rerank/text-rerank/text-rerank',BailianRerankSettings.endpoint+'?key=x']:
            self.assertFalse(b.validate_endpoint(url))


class BailianIntegrationTests(unittest.TestCase):
    setUp=fixtures.RetrievalEnhancementTests.setUp
    run_search=fixtures.RetrievalEnhancementTests.run_search

    def test_bailian_without_generation_provider_and_metadata_preserved(self):
        self.llm=None
        with patch.dict(os.environ,{'RETRIEVAL_RERANK_PROVIDER':'bailian'}),patch.object(b,'rerank',return_value=[(1,.9),(0,.2)]):
            value,_=self.run_search(False,True)
        self.assertEqual([x.code_chunk_id for x in value.results],[x.code_chunk_id for x in reversed(self.items[:2])])
        for x in value.results:
            original=next(i for i in self.items if i.code_chunk_id==x.code_chunk_id)
            self.assertEqual({k:v for k,v in asdict(x).items() if k!='rerank_rank'},
                             {k:v for k,v in asdict(original).items() if k!='rerank_rank'})
        self.assertEqual(value.audit['enhancement']['rerank']['status'],'succeeded')

    def test_all_bailian_failures_keep_original_order_and_reason(self):
        for reason in ['rerank_timeout','rerank_auth_failed','rerank_missing_items','rerank_duplicate_index','rerank_invalid_score']:
            with patch.dict(os.environ,{'RETRIEVAL_RERANK_PROVIDER':'bailian'}),patch.object(b,'rerank',side_effect=ProviderError(reason,'fixed')):
                value,_=self.run_search(False,True)
            self.assertEqual([x.code_chunk_id for x in value.results],[x.code_chunk_id for x in self.items[:2]])
            self.assertEqual(value.audit['enhancement']['rerank']['reason'],reason)
            self.assertEqual(value.audit['enhancement']['rerank']['status'],'fallback')

    def test_disabled_path_does_not_call_bailian(self):
        with patch.dict(os.environ,{'RETRIEVAL_RERANK_PROVIDER':'bailian'}),patch.object(b,'rerank') as mocked:
            value,args=self.run_search(False,False)
        mocked.assert_not_called();self.assertIs(value,self.base)

    def test_common_excerpt_and_configuration_independence(self):
        before=(get_llm_settings(),get_agent_limits())
        with patch.dict(os.environ,{'RETRIEVAL_RERANK_PROVIDER':'bailian','DASHSCOPE_API_KEY':'bailian-only'}):
            self.assertEqual((get_llm_settings(),get_agent_limits()),before)
        payload=candidate_payload('original question',self.items[:2]);captured={}
        def rank(q,documents,**kw):captured.update(q=q,docs=[json.loads(x) for x in documents]);return [(0,.5),(1,.4)]
        with patch.object(b,'rerank',side_effect=rank):rerank_candidates('original question',self.items[:2],provider='bailian',llm=self.llm,deadline=time.monotonic()+1)
        self.assertEqual(captured['docs'],payload['candidates']);self.assertEqual(captured['q'],payload['question'])

    def test_safe_history_retains_provider_and_numeric_error_not_key(self):
        safe=sanitize_retrieval_enhancement({'rerank':{'status':'fallback','reason':'rerank_http_503','provider':'bailian','api_key':'never-save',
            'excerpt_policy':'source-prefix-1600-characters-v1','truncated_candidate_ids':[1,True,'x'],
            'scores':[{'id':1,'index':0,'relevance_score':.3},{'id':2,'index':1,'relevance_score':float('nan')}]}})
        self.assertEqual(safe['rerank']['reason'],'rerank_http_503');self.assertEqual(safe['rerank']['provider'],'bailian')
        self.assertNotIn('never-save',str(safe))
        self.assertEqual(safe['rerank']['truncated_candidate_ids'],[1])
        self.assertEqual(safe['rerank']['scores'],[{'id':1,'index':0,'relevance_score':.3}])
