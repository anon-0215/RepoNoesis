// @vitest-environment jsdom
import React,{act} from 'react';
import {createRoot} from 'react-dom/client';
import {describe,it,expect} from 'vitest';
import {CodeViewer} from './CodeViewer';
import {sourceLines} from '../lib/sourceLines';
(globalThis as {IS_REACT_ACT_ENVIRONMENT?:boolean}).IS_REACT_ACT_ENVIRONMENT=true;
describe('physical source statement highlighting',()=>{
 for(const separator of ['\v','\f','\x1c','\x1d','\x1e','\x85','\u2028','\u2029'])
  for(const newline of ['\n','\r\n','\r'])
   it(`keeps statement coordinates ${JSON.stringify(separator)} ${JSON.stringify(newline)}`,async()=>{
    const rows=[`value="left${separator}right"`,`# note${separator}same comment`,'def before(): return 0','def after():','    raise ValueError("neutral")','def test_after():','    assert after() is None'];
    for(const ending of ['',newline]){
     const content=rows.join(newline)+ending;expect(sourceLines(content)).toEqual(rows);
     const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
     try{
      for(const line of [5,7]){
       await act(async()=>root.render(<CodeViewer key={line} source={{code:content,path:'neutral.py',label:'source'}} fullFile={{start:line,end:line}}/>));
       const selected=host.querySelectorAll('.sf-line.is-cited');expect(selected).toHaveLength(1);expect(selected[0].getAttribute('data-source-line')).toBe(String(line));expect(selected[0].querySelector('.sf-line-text')?.textContent).toBe(rows[line-1]);
       expect(host.querySelector('[data-source-line="1"] .sf-line-text')?.textContent).toBe(rows[0]);
      }
     }finally{await act(async()=>root.unmount());host.remove();}
    }
   });
});
