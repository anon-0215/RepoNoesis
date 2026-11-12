// Uses the existing frontend jsdom dependency; no generated build or Provider.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {JSDOM}=require('jsdom');
const html=fs.readFileSync(path.join(__dirname,'shell.html'),'utf8');
const settle=()=>new Promise(resolve=>setTimeout(resolve,15));
async function fixture(flags={}) {
  let saved={model_directory:'model',git_directory:'git',api_base:'',api_model:'same-model',
    data_directory:'test-data',key_saved:true,device:'auto',batch_size:1,max_length:8192,...flags};
  let generation=1,fail=false,reads=0,saves=0;
  const api={get_settings:async()=>{reads++;return {...saved};},get_status:async()=>({phase:'ready',generation,
    api_ready:true,backend_url:'http://localhost:18771',message:'ready'}),save_settings:async draft=>{
      saves++;if(fail)return {ok:false,message:'未保存：正在生成回答，请结束后再保存。'};
      saved={...saved,...draft};return {ok:true,settings:{...saved},changed:true};
    }};
  const dom=new JSDOM(html,{runScripts:'dangerously',url:'http://localhost/shell.html',
    beforeParse(window){window.pywebview={api};}});
  dom.window.dispatchEvent(new dom.window.Event('pywebviewready'));await settle();
  return {dom,api,get saved(){return saved;},get reads(){return reads;},get saves(){return saves;},
    fail:()=>fail=true,next:()=>generation++};
}
for(const hyde of [false,true])for(const rerank of [false,true])test(`read/save/reopen ${hyde}/${rerank}`,async()=>{
  const f=await fixture({hyde_enabled:hyde,rerank_enabled:rerank});
  try{
    const w=f.dom.window,d=w.document;
    await w.showSettings();
    assert.equal(d.getElementById('hyde-enabled').checked,hyde);
    assert.equal(d.getElementById('rerank-enabled').checked,rerank);
    d.getElementById('hyde-enabled').click();
    assert.equal(d.getElementById('rerank-enabled').checked,rerank);
    await d.getElementById('settings-form').onsubmit({preventDefault(){}});
    assert.equal(f.saved.hyde_enabled,!hyde);assert.equal(f.saved.rerank_enabled,rerank);
    assert.equal(typeof f.saved.hyde_enabled,'boolean');
    assert.equal(d.getElementById('save-message').dataset.applying,'true'); // Ready old generation is not new config.
    f.next();await w.poll();
    assert.equal(d.getElementById('save-message').dataset.applying,'');
    assert.equal(d.getElementById('settings').hidden,true);
    await w.showSettings();assert.equal(d.getElementById('hyde-enabled').checked,!hyde);
    assert.equal(d.getElementById('api-model').value,'same-model');
    assert.equal(d.getElementById('api-key').value,'');assert.equal(f.saves,1);assert.ok(f.reads>=3);
    assert.match(d.getElementById('hyde-enabled').getAttribute('aria-describedby'),/hyde-help/);
  }finally{f.dom.window.close();}
});
test('failed save never announces applied state and reopen reads saved truth',async()=>{
  const f=await fixture({hyde_enabled:false,rerank_enabled:false});
  try{
    const w=f.dom.window,d=w.document;await w.showSettings();f.fail();
    d.getElementById('hyde-enabled').click();
    await d.getElementById('settings-form').onsubmit({preventDefault(){}});
    assert.equal(f.saved.hyde_enabled,false);assert.equal(d.getElementById('save-message').dataset.applying,'');
    assert.match(d.getElementById('save-message').textContent,/未保存/);
    await w.showSettings();assert.equal(d.getElementById('hyde-enabled').checked,false);
  }finally{f.dom.window.close();}
});
test('invalid legacy strings are unchecked and load failure prevents saving',async()=>{
  const f=await fixture({hyde_enabled:'false',rerank_enabled:'true'});
  try{
    const w=f.dom.window,d=w.document;assert.equal(d.getElementById('hyde-enabled').checked,false);
    assert.equal(d.getElementById('rerank-enabled').checked,false);
    f.api.get_settings=async()=>{throw new Error('fixture');};await w.showSettings();
    assert.equal(d.getElementById('save-settings').disabled,true);
    await d.getElementById('settings-form').onsubmit({preventDefault(){}});assert.equal(f.saves,0);
  }finally{f.dom.window.close();}
});
