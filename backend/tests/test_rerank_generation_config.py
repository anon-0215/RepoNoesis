from dataclasses import asdict
import io,json,os,time,unittest
import urllib.error
from unittest.mock import patch
from app.config import LLMSettings,get_retrieval_enhancement_settings
from app.services.llm_client import LLMClient,ProviderError
from app.services.retrieval_enhancement import rerank_candidates
from tests import test_retrieval_enhancement as fixtures


class GenerationRerankTests(unittest.TestCase):
    setUp=fixtures.RetrievalEnhancementTests.setUp
    run_search=fixtures.RetrievalEnhancementTests.run_search

    def test_generic_model_parameters_and_shared_client_unchanged(self):
        settings=LLMSettings(provider='openai_compatible',base_url='https://generic.invalid/v1',api_key='fixture-key',
            model='base-generic-model',answer_model='answer-only-model',planner_max_generation_tokens=8192,
            answer_max_generation_tokens=8192,answer_reasoning_effort='high',max_retries=0)
        requests=[]
        def opener(request,**kw):
            payload=json.loads(request.data);requests.append(payload)
            content=json.dumps({'ids':[x.code_chunk_id for x in self.items[:2]]}) if payload['model']=='base-generic-model' else 'answer'
            if payload['model']=='base-generic-model':
                self.assertNotIn('thinking',payload);self.assertNotIn('reasoning_effort',payload)
                self.assertEqual(payload['max_tokens'],240);self.assertEqual(payload['temperature'],0)
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':content}}]}).encode())
        client=LLMClient(settings,opener=opener);before=asdict(client.settings)
        audit={};rerank_candidates('question',self.items[:2],provider='generation',llm=client,deadline=time.monotonic()+2,audit=audit)
        client.chat([{'role':'user','content':'answer'}],purpose='final_answer')
        self.assertEqual(requests[-1]['model'],'answer-only-model');self.assertEqual(requests[-1]['max_tokens'],8192)
        self.assertEqual(requests[-1]['reasoning_effort'],'high')
        self.assertEqual(client.model,'base-generic-model');self.assertEqual(asdict(client.settings),before)
        self.assertEqual(audit['provider'],'generation');self.assertEqual(audit['model'],'base-generic-model')

    def test_configuration_aliases_and_explicit_mode(self):
        for env,expected in [({},'generation'),({'RETRIEVAL_RERANK_PROVIDER':'deepseek'},'generation'),
                             ({'RETRIEVAL_RERANK_PROVIDER':'bailian'},'bailian'),
                             ({'RETRIEVAL_RERANK_PROVIDER':'bailian','RETRIEVAL_RERANK_MODE':'generation'},'generation'),
                             ({'RETRIEVAL_RERANK_MODE':'independent','RETRIEVAL_RERANK_SERVICE':'bailian'},'bailian'),
                             ({'RETRIEVAL_RERANK_MODE':'invalid'},'invalid')]:
            with patch.dict(os.environ,env,clear=True):self.assertEqual(get_retrieval_enhancement_settings().rerank_provider,expected)

    def test_default_error_preserves_base_and_never_switches_provider(self):
        self.llm.chat=lambda *a,**k:(_ for _ in ()).throw(ProviderError('provider_authentication_failed','fixed'))
        with patch.dict(os.environ,{'RETRIEVAL_RERANK_MODE':'generation'}),patch('app.services.bailian_reranker.rerank') as bailian:
            value,_=self.run_search(False,True)
        bailian.assert_not_called();self.assertEqual([x.code_chunk_id for x in value.results],[x.code_chunk_id for x in self.items[:2]])
        self.assertEqual(value.audit['enhancement']['rerank']['status'],'fallback')
        self.assertEqual(value.audit['enhancement']['rerank']['provider'],'generation')

    def test_closed_requests_neither_provider(self):
        for mode in ['generation','independent']:
            with patch.dict(os.environ,{'RETRIEVAL_RERANK_MODE':mode}),patch('app.services.bailian_reranker.rerank') as bailian:
                value,_=self.run_search(False,False)
            bailian.assert_not_called();self.assertEqual(self.calls,[]);self.assertIs(value,self.base)

    def test_failed_rerank_cannot_change_later_answer_model_or_parameters(self):
        settings=LLMSettings(provider='openai_compatible',base_url='https://generic.invalid/v1',api_key='fixture-key',
            model='generic-base',answer_model='separate-answer',answer_max_generation_tokens=4096,max_retries=0)
        payloads=[]
        def opener(request,**kw):
            payload=json.loads(request.data);payloads.append(payload)
            if payload['model']=='generic-base':
                raise urllib.error.HTTPError(request.full_url,401,'fixed',{},None)
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':'answer'}}]}).encode())
        client=LLMClient(settings,opener=opener)
        with self.assertRaises(ProviderError):
            rerank_candidates('question',self.items[:2],provider='generation',llm=client,deadline=time.monotonic()+2)
        client.chat([{'role':'user','content':'answer'}],purpose='final_answer')
        self.assertEqual([p['model'] for p in payloads],['generic-base','separate-answer'])
        self.assertEqual(payloads[-1]['max_tokens'],4096)
        self.assertEqual(client.model,'generic-base');self.assertEqual(client.settings,settings)
