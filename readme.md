# RASA框架解析

## 1、Rasa重要类和方法导入


```python
import os

from rasa.importers.importer import TrainingDataImporter #训练数据加载

from rasa.utils.io import pickle_dump,json_pickle        
from rasa.utils.common import TempDirectoryPath          #数据零时目录路径
from rasa.utils.endpoints import read_endpoint_config    #action_endpoint，endpoint文件读取

from rasa.nlu import components, utils                   #nlu组件模块
from rasa.nlu import config as nlu_config                #nlu config模块
from rasa.nlu.model import Interpreter, Trainer          #nlu模型解析器，训练器接口
from rasa.nlu.train import train as NLU_train            #nlu训练接口
from rasa.nlu.config import RasaNLUModelConfig           #nlu模型config接口
from rasa.nlu.training_data import Message
from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

from rasa.core import config as core_config              #core config模块
from rasa.core.domain import Domain                      #core domain接口
from rasa.core.agent import Agent,load_agent             #core Agent接口
from rasa.core.interpreter import RasaNLUInterpreter     #core nlu解析器接口
from rasa.core.channels.channel import UserMessage       #core 用户消息接口
from rasa.core.channels import CollectingOutputChannel   #core 输出消息收集接口

from rasa.core.featurizers import BinarySingleStateFeaturizer
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer

from rasa.train import (
    train, train_async, 
    _train_nlu_with_validated_data    #rasa 模型训练上层接口
)

from parsing_utils import json_dump
```

## 2、Rasa项目的原始数据导入

一个Rasa项目，主要数据文件包括：nlu.md, stories.md, domain.yml, config.yml, endpoints.yml, credentials.yml等等。下面导入其中几个关键数据文件：nlu.md, stories.md, domain.yml, config.yml。

### 2.1、metadata_parsing


```python
domain_file = 'domain.yml'
configs_file = 'config_15.yml'
training_files = 'data'

output_path = 'models/model_15'
model_path = 'models/models_15'

file_importer = TrainingDataImporter.load_from_config(configs_file,domain_file,
                                             training_files)
file_importer.__dict__['_importers'][0].__dict__
```




    {'config': {'language': 'zh',
      'pipeline': [{'name': 'JiebaTokenizer'},
       {'name': 'KashgariEntityExtractor',
        'bert_model_path': 'chinese_L-12_H-768_A-12'},
       {'name': 'RegexExtractor', 'regex_ner_entity': ['EqpType']},
       {'name': 'KashgariIntentClassifier',
        'bert_model_path': 'chinese_L-12_H-768_A-12'}],
      'policies': [{'name': 'TwoStageFallbackPolicy',
        'core_threshold': 0.1,
        'nlu_threshold': 0.1,
        'fallback_core_action_name': 'action_default_fallback',
        'fallback_nlu_action_name': 'action_handoff_to_human',
        'deny_suggestion_intent_name': 'out_of_scope'},
       {'name': 'MemoizationPolicy', 'max_history': 3},
       {'name': 'FormPolicy'},
       {'name': 'MappingPolicy'},
       {'name': 'KerasPolicy',
        'featurizer': [{'name': 'MaxHistoryTrackerFeaturizer',
          'max_history': 5,
          'state_featurizer': [{'name': 'BinarySingleStateFeaturizer'}]}]}]},
     '_domain_path': 'domain.yml',
     '_story_files': ['data\\stories.md'],
     '_nlu_files': ['data\\nlu.md']}



返回类 rasa.nlu.training_data.training_data.TrainingData


```python
domain = await file_importer.get_domain() 
#domain.__dict__
domain
```




    <rasa.core.domain.Domain at 0x1a3bf72cf08>



stories.md文件，被解析成story_steps：它主要由StoryStep列表组成，一条StoryStep对应一个story，events代表每个story中bot和用户产生的对话动作
stories_data.as_story_string


```python
stories_data = await file_importer.get_stories() #返回类 rasa.nlu.training_data.training_data.TrainingData
stories_data.__dict__
```




    {'story_steps': [StoryStep(block_name='equipFaultDiag_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF73A248>, <rasa.core.events.ActionExecuted object at 0x000001A3BF6B41C8>, <rasa.core.events.Form object at 0x000001A3BF6ABFC8>, <rasa.core.events.Form object at 0x000001A3BF26CB88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF2646C8>]),
      StoryStep(block_name='sysFaultDiag_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF7259C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF73A4C8>, <rasa.core.events.Form object at 0x000001A3BDB57D48>, <rasa.core.events.Form object at 0x000001A3BF725848>, <rasa.core.events.ActionExecuted object at 0x000001A3BF7208C8>]),
      StoryStep(block_name='equipFaultDiag Generated Form Story 1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF725808>, <rasa.core.events.ActionExecuted object at 0x000001A3BF705C88>, <rasa.core.events.Form object at 0x000001A3BF705148>, <rasa.core.events.SlotSet object at 0x000001A3BF705248>, <rasa.core.events.SlotSet object at 0x000001A3BF8C1BC8>, <rasa.core.events.SlotSet object at 0x000001A3BF7AB808>, <rasa.core.events.SlotSet object at 0x000001A3BF7ABF48>, <rasa.core.events.SlotSet object at 0x000001A3BF7AB708>, <rasa.core.events.SlotSet object at 0x000001A3BF7ABA48>, <rasa.core.events.SlotSet object at 0x000001A3BF7AB788>, <rasa.core.events.SlotSet object at 0x000001A3BF857808>, <rasa.core.events.SlotSet object at 0x000001A3BF7154C8>, <rasa.core.events.Form object at 0x000001A3BF7A8108>, <rasa.core.events.SlotSet object at 0x000001A3BF7A2408>, <rasa.core.events.ActionExecuted object at 0x000001A3BF7A2F08>]),
      StoryStep(block_name='equipFaultDiag Generated Form Story 2', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF7A2F48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF7A25C8>, <rasa.core.events.Form object at 0x000001A3BF7A27C8>, <rasa.core.events.UserUttered object at 0x000001A3BF87D348>, <rasa.core.events.SlotSet object at 0x000001A3BF7A2D48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF7A2308>, <rasa.core.events.UserUttered object at 0x000001A3BF70A308>, <rasa.core.events.SlotSet object at 0x000001A3BF70F8C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BC336048>, <rasa.core.events.SlotSet object at 0x000001A3BF7A7508>, <rasa.core.events.UserUttered object at 0x000001A3BF6F1648>, <rasa.core.events.SlotSet object at 0x000001A3BF7A7448>, <rasa.core.events.ActionExecuted object at 0x000001A3BF6F1908>, <rasa.core.events.SlotSet object at 0x000001A3BF852B48>, <rasa.core.events.UserUttered object at 0x000001A3BF852608>, <rasa.core.events.SlotSet object at 0x000001A3BF852D08>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8526C8>, <rasa.core.events.Form object at 0x000001A3BF7A1F48>, <rasa.core.events.SlotSet object at 0x000001A3BF8A29C8>]),
      StoryStep(block_name='equipFaultDiag Generated Form Story 3', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8B9988>, <rasa.core.events.SlotSet object at 0x000001A3BF877508>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9608>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9748>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9888>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8B9948>, <rasa.core.events.Form object at 0x000001A3BF8B9CC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8B99C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9D48>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9D88>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9E08>, <rasa.core.events.Form object at 0x000001A3BF8B9E88>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9F88>]),
      StoryStep(block_name='equipFaultDiag Generated Form Story 4', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8AB1C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9FC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB2C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB348>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8AB3C8>, <rasa.core.events.Form object at 0x000001A3BF8AB508>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB448>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB588>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB5C8>, <rasa.core.events.UserUttered object at 0x000001A3BF8AB6C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8AB648>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB848>, <rasa.core.events.Form object at 0x000001A3BF8AB7C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB908>]),
      StoryStep(block_name='equipFaultDiag Generated Form Story 5', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8ABA48>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB948>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABB48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8ABBC8>, <rasa.core.events.Form object at 0x000001A3BF8ABC88>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABC08>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABD08>, <rasa.core.events.UserUttered object at 0x000001A3BF8ABEC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABD88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8ABF08>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3048>, <rasa.core.events.UserUttered object at 0x000001A3BF8D3088>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABFC8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D31C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3308>, <rasa.core.events.Form object at 0x000001A3BF8D3288>, <rasa.core.events.SlotSet object at 0x000001A3BF8D33C8>]),
      StoryStep(block_name='equipMaintenance path_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8AB8C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3408>, <rasa.core.events.Form object at 0x000001A3BF8D3588>, <rasa.core.events.Form object at 0x000001A3BF8D34C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3648>]),
      StoryStep(block_name='sysMaintenance_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D3548>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3748>, <rasa.core.events.Form object at 0x000001A3BF8D3948>, <rasa.core.events.Form object at 0x000001A3BF8D37C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3A08>]),
      StoryStep(block_name='equipMaintenance Generated Form Story 1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D38C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3A48>, <rasa.core.events.Form object at 0x000001A3BF8D3CC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3B48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3D48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3DC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3E48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3F08>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3FC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6088>, <rasa.core.events.Form object at 0x000001A3BF8D6108>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6208>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D6248>]),
      StoryStep(block_name='equipMaintenance Generated Form Story 2', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8B9F48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D6308>, <rasa.core.events.Form object at 0x000001A3BF8D6588>, <rasa.core.events.UserUttered object at 0x000001A3BF8D6688>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6408>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D66C8>, <rasa.core.events.UserUttered object at 0x000001A3BF8D6848>, <rasa.core.events.SlotSet object at 0x000001A3BF8D68C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D6888>, <rasa.core.events.SlotSet object at 0x000001A3BF8D67C8>, <rasa.core.events.UserUttered object at 0x000001A3BF8D6B08>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6988>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D6B48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6D08>, <rasa.core.events.Form object at 0x000001A3BF8D6C48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6DC8>]),
      StoryStep(block_name='equipMaintenance Generated Form Story 3', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D6F48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6E08>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8088>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8208>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D82C8>, <rasa.core.events.Form object at 0x000001A3BF8D8408>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8348>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8488>, <rasa.core.events.Form object at 0x000001A3BF8D84C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8588>]),
      StoryStep(block_name='equipMaintenance Generated Form Story 4', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D86C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D85C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D87C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8948>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D8A08>, <rasa.core.events.Form object at 0x000001A3BF8D8B48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8A88>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8BC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8C08>, <rasa.core.events.Form object at 0x000001A3BF8D8C88>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8B08>]),
      StoryStep(block_name='equipMaintenance Generated Form Story 5', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D8E48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8D88>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8F48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DF108>, <rasa.core.events.Form object at 0x000001A3BF8DF208>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF188>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF288>, <rasa.core.events.UserUttered object at 0x000001A3BF8DF448>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF308>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DF488>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF608>, <rasa.core.events.Form object at 0x000001A3BF8DF588>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF6C8>]),
      StoryStep(block_name='equipOperation path_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8DF7C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DF708>, <rasa.core.events.Form object at 0x000001A3BF8DFA88>, <rasa.core.events.Form object at 0x000001A3BF8DF908>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DFB48>]),
      StoryStep(block_name='sysOperation_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8DFB88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DF688>, <rasa.core.events.Form object at 0x000001A3BF8DFEC8>, <rasa.core.events.Form object at 0x000001A3BF8DFD48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DFF88>]),
      StoryStep(block_name='equipOperation Generated Form Story 1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E1048>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E10C8>, <rasa.core.events.Form object at 0x000001A3BF8E1348>, <rasa.core.events.SlotSet object at 0x000001A3BF8E11C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E13C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1448>, <rasa.core.events.SlotSet object at 0x000001A3BF8E14C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1588>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1648>, <rasa.core.events.SlotSet object at 0x000001A3BF8E16C8>, <rasa.core.events.Form object at 0x000001A3BF8E1788>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1848>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E1888>]),
      StoryStep(block_name='equipOperation Generated Form Story 2', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E1948>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E19C8>, <rasa.core.events.Form object at 0x000001A3BF8E1C48>, <rasa.core.events.UserUttered object at 0x000001A3BF8E1D48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1AC8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E1D88>, <rasa.core.events.UserUttered object at 0x000001A3BF8E1F08>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1F88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E1F48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5088>, <rasa.core.events.UserUttered object at 0x000001A3BF8E5208>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1E88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E5248>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5408>, <rasa.core.events.Form object at 0x000001A3BF8E5348>, <rasa.core.events.SlotSet object at 0x000001A3BF8E54C8>]),
      StoryStep(block_name='equipOperation Generated Form Story 3', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E5688>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5508>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5788>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5908>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E59C8>, <rasa.core.events.Form object at 0x000001A3BF8E5B08>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5A48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5B88>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5BC8>, <rasa.core.events.Form object at 0x000001A3BF8E5C48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5AC8>]),
      StoryStep(block_name='equipOperation Generated Form Story 4', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E5EC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5D48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5FC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7188>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E7248>, <rasa.core.events.Form object at 0x000001A3BF8E7388>, <rasa.core.events.SlotSet object at 0x000001A3BF8E72C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7408>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7448>, <rasa.core.events.Form object at 0x000001A3BF8E74C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7348>]),
      StoryStep(block_name='equipOperation Generated Form Story 5', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E76C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E75C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E77C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E7948>, <rasa.core.events.Form object at 0x000001A3BF8E7A48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E79C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7AC8>, <rasa.core.events.UserUttered object at 0x000001A3BF8E7C88>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7B48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E7CC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7E48>, <rasa.core.events.Form object at 0x000001A3BF8E7DC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7F08>]),
      StoryStep(block_name='equipInstall path_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EB088>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E7F48>, <rasa.core.events.Form object at 0x000001A3BF8EB308>, <rasa.core.events.Form object at 0x000001A3BF8EB188>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB3C8>]),
      StoryStep(block_name='sysInstall_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EB408>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB508>, <rasa.core.events.Form object at 0x000001A3BF8EB7C8>, <rasa.core.events.Form object at 0x000001A3BF8EB608>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB888>]),
      StoryStep(block_name='equipInstall Generated Form Story 1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EB8C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB388>, <rasa.core.events.Form object at 0x000001A3BF8EBC08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBA88>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBC88>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBD08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBD88>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBE48>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBF08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBF88>, <rasa.core.events.Form object at 0x000001A3BF8EE048>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE148>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EE188>]),
      StoryStep(block_name='equipInstall Generated Form Story 2', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EE208>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB848>, <rasa.core.events.Form object at 0x000001A3BF8EE548>, <rasa.core.events.UserUttered object at 0x000001A3BF8EE648>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE3C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EE688>, <rasa.core.events.UserUttered object at 0x000001A3BF8EE808>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE888>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EE848>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE788>, <rasa.core.events.UserUttered object at 0x000001A3BF8EEAC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE948>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EEB08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EECC8>, <rasa.core.events.Form object at 0x000001A3BF8EEC08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EED88>]),
      StoryStep(block_name='equipInstall Generated Form Story 3', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EEF48>, <rasa.core.events.SlotSet object at 0x000001A3BF8EEDC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1088>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1208>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8F12C8>, <rasa.core.events.Form object at 0x000001A3BF8F1408>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1348>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1488>, <rasa.core.events.SlotSet object at 0x000001A3BF8F14C8>, <rasa.core.events.Form object at 0x000001A3BF8F1548>, <rasa.core.events.SlotSet object at 0x000001A3BF8F13C8>]),
      StoryStep(block_name='equipInstall Generated Form Story 4', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8F17C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1648>, <rasa.core.events.SlotSet object at 0x000001A3BF8F18C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1A48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8F1B08>, <rasa.core.events.Form object at 0x000001A3BF8F1C48>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1B88>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1CC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1D08>, <rasa.core.events.Form object at 0x000001A3BF8F1D88>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1C08>]),
      StoryStep(block_name='equipInstall Generated Form Story 5', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8F1F88>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1E88>, <rasa.core.events.SlotSet object at 0x000001A3BF8F50C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8F5248>, <rasa.core.events.Form object at 0x000001A3BF8F5348>, <rasa.core.events.SlotSet object at 0x000001A3BF8F52C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F53C8>, <rasa.core.events.UserUttered object at 0x000001A3BF70A108>, <rasa.core.events.SlotSet object at 0x000001A3BF70AB08>, <rasa.core.events.ActionExecuted object at 0x000001A3BF70A608>, <rasa.core.events.SlotSet object at 0x000001A3BF733748>, <rasa.core.events.Form object at 0x000001A3BF8BC808>, <rasa.core.events.SlotSet object at 0x000001A3BF87D5C8>]),
      StoryStep(block_name='deviceParameter path_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8F1E48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF87D508>, <rasa.core.events.Form object at 0x000001A3BF876E88>, <rasa.core.events.Form object at 0x000001A3BF876E48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF6F1BC8>])],
     'step_lookup': {'30_59cb32cb69a14425b5ec7ceb21dfab88': StoryStep(block_name='equipFaultDiag_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF73A248>, <rasa.core.events.ActionExecuted object at 0x000001A3BF6B41C8>, <rasa.core.events.Form object at 0x000001A3BF6ABFC8>, <rasa.core.events.Form object at 0x000001A3BF26CB88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF2646C8>]),
      '31_5436161f1ba141d79006debc4e61a004': StoryStep(block_name='sysFaultDiag_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF7259C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF73A4C8>, <rasa.core.events.Form object at 0x000001A3BDB57D48>, <rasa.core.events.Form object at 0x000001A3BF725848>, <rasa.core.events.ActionExecuted object at 0x000001A3BF7208C8>]),
      '32_6ab17da8458f44db8c4f0d809fb92cc6': StoryStep(block_name='equipFaultDiag Generated Form Story 1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF725808>, <rasa.core.events.ActionExecuted object at 0x000001A3BF705C88>, <rasa.core.events.Form object at 0x000001A3BF705148>, <rasa.core.events.SlotSet object at 0x000001A3BF705248>, <rasa.core.events.SlotSet object at 0x000001A3BF8C1BC8>, <rasa.core.events.SlotSet object at 0x000001A3BF7AB808>, <rasa.core.events.SlotSet object at 0x000001A3BF7ABF48>, <rasa.core.events.SlotSet object at 0x000001A3BF7AB708>, <rasa.core.events.SlotSet object at 0x000001A3BF7ABA48>, <rasa.core.events.SlotSet object at 0x000001A3BF7AB788>, <rasa.core.events.SlotSet object at 0x000001A3BF857808>, <rasa.core.events.SlotSet object at 0x000001A3BF7154C8>, <rasa.core.events.Form object at 0x000001A3BF7A8108>, <rasa.core.events.SlotSet object at 0x000001A3BF7A2408>, <rasa.core.events.ActionExecuted object at 0x000001A3BF7A2F08>]),
      '33_ddbbf2fcb04745428807e5f0591ff77c': StoryStep(block_name='equipFaultDiag Generated Form Story 2', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF7A2F48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF7A25C8>, <rasa.core.events.Form object at 0x000001A3BF7A27C8>, <rasa.core.events.UserUttered object at 0x000001A3BF87D348>, <rasa.core.events.SlotSet object at 0x000001A3BF7A2D48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF7A2308>, <rasa.core.events.UserUttered object at 0x000001A3BF70A308>, <rasa.core.events.SlotSet object at 0x000001A3BF70F8C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BC336048>, <rasa.core.events.SlotSet object at 0x000001A3BF7A7508>, <rasa.core.events.UserUttered object at 0x000001A3BF6F1648>, <rasa.core.events.SlotSet object at 0x000001A3BF7A7448>, <rasa.core.events.ActionExecuted object at 0x000001A3BF6F1908>, <rasa.core.events.SlotSet object at 0x000001A3BF852B48>, <rasa.core.events.UserUttered object at 0x000001A3BF852608>, <rasa.core.events.SlotSet object at 0x000001A3BF852D08>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8526C8>, <rasa.core.events.Form object at 0x000001A3BF7A1F48>, <rasa.core.events.SlotSet object at 0x000001A3BF8A29C8>]),
      '34_e543523258f14563926bc7934c17b936': StoryStep(block_name='equipFaultDiag Generated Form Story 3', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8B9988>, <rasa.core.events.SlotSet object at 0x000001A3BF877508>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9608>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9748>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9888>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8B9948>, <rasa.core.events.Form object at 0x000001A3BF8B9CC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8B99C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9D48>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9D88>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9E08>, <rasa.core.events.Form object at 0x000001A3BF8B9E88>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9F88>]),
      '35_4b422fd5799b48d5b0d857e862111ec3': StoryStep(block_name='equipFaultDiag Generated Form Story 4', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8AB1C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8B9FC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB2C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB348>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8AB3C8>, <rasa.core.events.Form object at 0x000001A3BF8AB508>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB448>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB588>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB5C8>, <rasa.core.events.UserUttered object at 0x000001A3BF8AB6C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8AB648>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB848>, <rasa.core.events.Form object at 0x000001A3BF8AB7C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB908>]),
      '36_92b0ebd96d034401bc6cf5e6631ee0aa': StoryStep(block_name='equipFaultDiag Generated Form Story 5', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8ABA48>, <rasa.core.events.SlotSet object at 0x000001A3BF8AB948>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABB48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8ABBC8>, <rasa.core.events.Form object at 0x000001A3BF8ABC88>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABC08>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABD08>, <rasa.core.events.UserUttered object at 0x000001A3BF8ABEC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABD88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8ABF08>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3048>, <rasa.core.events.UserUttered object at 0x000001A3BF8D3088>, <rasa.core.events.SlotSet object at 0x000001A3BF8ABFC8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D31C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3308>, <rasa.core.events.Form object at 0x000001A3BF8D3288>, <rasa.core.events.SlotSet object at 0x000001A3BF8D33C8>]),
      '37_32e443a972d34a6c93342ffa794248e0': StoryStep(block_name='equipMaintenance path_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8AB8C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3408>, <rasa.core.events.Form object at 0x000001A3BF8D3588>, <rasa.core.events.Form object at 0x000001A3BF8D34C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3648>]),
      '38_ff72b1e474eb4a85b76bf1a9cfa886e9': StoryStep(block_name='sysMaintenance_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D3548>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3748>, <rasa.core.events.Form object at 0x000001A3BF8D3948>, <rasa.core.events.Form object at 0x000001A3BF8D37C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3A08>]),
      '39_4ce9f23606444e6e8be2cbceb62f4eec': StoryStep(block_name='equipMaintenance Generated Form Story 1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D38C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D3A48>, <rasa.core.events.Form object at 0x000001A3BF8D3CC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3B48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3D48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3DC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3E48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3F08>, <rasa.core.events.SlotSet object at 0x000001A3BF8D3FC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6088>, <rasa.core.events.Form object at 0x000001A3BF8D6108>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6208>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D6248>]),
      '40_a171f49097f9400993a175980c61fc91': StoryStep(block_name='equipMaintenance Generated Form Story 2', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8B9F48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D6308>, <rasa.core.events.Form object at 0x000001A3BF8D6588>, <rasa.core.events.UserUttered object at 0x000001A3BF8D6688>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6408>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D66C8>, <rasa.core.events.UserUttered object at 0x000001A3BF8D6848>, <rasa.core.events.SlotSet object at 0x000001A3BF8D68C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D6888>, <rasa.core.events.SlotSet object at 0x000001A3BF8D67C8>, <rasa.core.events.UserUttered object at 0x000001A3BF8D6B08>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6988>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D6B48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6D08>, <rasa.core.events.Form object at 0x000001A3BF8D6C48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6DC8>]),
      '41_0881e034b025458983ecbc7a875abd25': StoryStep(block_name='equipMaintenance Generated Form Story 3', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D6F48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D6E08>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8088>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8208>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D82C8>, <rasa.core.events.Form object at 0x000001A3BF8D8408>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8348>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8488>, <rasa.core.events.Form object at 0x000001A3BF8D84C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8588>]),
      '42_2ac0edb631204c4fb0f5cc22da3f14ae': StoryStep(block_name='equipMaintenance Generated Form Story 4', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D86C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D85C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D87C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8948>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8D8A08>, <rasa.core.events.Form object at 0x000001A3BF8D8B48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8A88>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8BC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8C08>, <rasa.core.events.Form object at 0x000001A3BF8D8C88>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8B08>]),
      '43_acc7066f4e584ec487380350a14608bc': StoryStep(block_name='equipMaintenance Generated Form Story 5', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8D8E48>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8D88>, <rasa.core.events.SlotSet object at 0x000001A3BF8D8F48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DF108>, <rasa.core.events.Form object at 0x000001A3BF8DF208>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF188>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF288>, <rasa.core.events.UserUttered object at 0x000001A3BF8DF448>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF308>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DF488>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF608>, <rasa.core.events.Form object at 0x000001A3BF8DF588>, <rasa.core.events.SlotSet object at 0x000001A3BF8DF6C8>]),
      '44_554dd70f09f94f6aaa1bcd4122385a7a': StoryStep(block_name='equipOperation path_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8DF7C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DF708>, <rasa.core.events.Form object at 0x000001A3BF8DFA88>, <rasa.core.events.Form object at 0x000001A3BF8DF908>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DFB48>]),
      '45_72ec892318b848f4b7eaf96768b5ab60': StoryStep(block_name='sysOperation_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8DFB88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DF688>, <rasa.core.events.Form object at 0x000001A3BF8DFEC8>, <rasa.core.events.Form object at 0x000001A3BF8DFD48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8DFF88>]),
      '46_a96f4cf20fd041be82c0fbec6f1422bc': StoryStep(block_name='equipOperation Generated Form Story 1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E1048>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E10C8>, <rasa.core.events.Form object at 0x000001A3BF8E1348>, <rasa.core.events.SlotSet object at 0x000001A3BF8E11C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E13C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1448>, <rasa.core.events.SlotSet object at 0x000001A3BF8E14C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1588>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1648>, <rasa.core.events.SlotSet object at 0x000001A3BF8E16C8>, <rasa.core.events.Form object at 0x000001A3BF8E1788>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1848>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E1888>]),
      '47_9ced12c5cb7d4d0aaff4753e795e0b5c': StoryStep(block_name='equipOperation Generated Form Story 2', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E1948>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E19C8>, <rasa.core.events.Form object at 0x000001A3BF8E1C48>, <rasa.core.events.UserUttered object at 0x000001A3BF8E1D48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1AC8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E1D88>, <rasa.core.events.UserUttered object at 0x000001A3BF8E1F08>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1F88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E1F48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5088>, <rasa.core.events.UserUttered object at 0x000001A3BF8E5208>, <rasa.core.events.SlotSet object at 0x000001A3BF8E1E88>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E5248>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5408>, <rasa.core.events.Form object at 0x000001A3BF8E5348>, <rasa.core.events.SlotSet object at 0x000001A3BF8E54C8>]),
      '48_6f113e0986604e38a6c45f013236a895': StoryStep(block_name='equipOperation Generated Form Story 3', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E5688>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5508>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5788>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5908>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E59C8>, <rasa.core.events.Form object at 0x000001A3BF8E5B08>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5A48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5B88>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5BC8>, <rasa.core.events.Form object at 0x000001A3BF8E5C48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5AC8>]),
      '49_9da19de845dd42c896098e07018d122a': StoryStep(block_name='equipOperation Generated Form Story 4', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E5EC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5D48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E5FC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7188>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E7248>, <rasa.core.events.Form object at 0x000001A3BF8E7388>, <rasa.core.events.SlotSet object at 0x000001A3BF8E72C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7408>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7448>, <rasa.core.events.Form object at 0x000001A3BF8E74C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7348>]),
      '50_3bee6a16c39c4eabaa1278f9bd96cfc2': StoryStep(block_name='equipOperation Generated Form Story 5', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8E76C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E75C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E77C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E7948>, <rasa.core.events.Form object at 0x000001A3BF8E7A48>, <rasa.core.events.SlotSet object at 0x000001A3BF8E79C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7AC8>, <rasa.core.events.UserUttered object at 0x000001A3BF8E7C88>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7B48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E7CC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7E48>, <rasa.core.events.Form object at 0x000001A3BF8E7DC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8E7F08>]),
      '51_20ad94aadf594d179b405c4903357db7': StoryStep(block_name='equipInstall path_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EB088>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8E7F48>, <rasa.core.events.Form object at 0x000001A3BF8EB308>, <rasa.core.events.Form object at 0x000001A3BF8EB188>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB3C8>]),
      '52_3825abe773fc4daca3da45e135d04e4d': StoryStep(block_name='sysInstall_form_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EB408>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB508>, <rasa.core.events.Form object at 0x000001A3BF8EB7C8>, <rasa.core.events.Form object at 0x000001A3BF8EB608>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB888>]),
      '53_37acd2b8d2a34cdba5e139047e57b358': StoryStep(block_name='equipInstall Generated Form Story 1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EB8C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB388>, <rasa.core.events.Form object at 0x000001A3BF8EBC08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBA88>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBC88>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBD08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBD88>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBE48>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBF08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EBF88>, <rasa.core.events.Form object at 0x000001A3BF8EE048>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE148>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EE188>]),
      '54_05045c417be746e89c91eac0af345766': StoryStep(block_name='equipInstall Generated Form Story 2', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EE208>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EB848>, <rasa.core.events.Form object at 0x000001A3BF8EE548>, <rasa.core.events.UserUttered object at 0x000001A3BF8EE648>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE3C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EE688>, <rasa.core.events.UserUttered object at 0x000001A3BF8EE808>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE888>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EE848>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE788>, <rasa.core.events.UserUttered object at 0x000001A3BF8EEAC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8EE948>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8EEB08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EECC8>, <rasa.core.events.Form object at 0x000001A3BF8EEC08>, <rasa.core.events.SlotSet object at 0x000001A3BF8EED88>]),
      '55_f6bc73cec1d44f4db2670a88fd2da682': StoryStep(block_name='equipInstall Generated Form Story 3', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8EEF48>, <rasa.core.events.SlotSet object at 0x000001A3BF8EEDC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1088>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1208>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8F12C8>, <rasa.core.events.Form object at 0x000001A3BF8F1408>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1348>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1488>, <rasa.core.events.SlotSet object at 0x000001A3BF8F14C8>, <rasa.core.events.Form object at 0x000001A3BF8F1548>, <rasa.core.events.SlotSet object at 0x000001A3BF8F13C8>]),
      '56_ecc468847db04cf58456cb7d6e916fa3': StoryStep(block_name='equipInstall Generated Form Story 4', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8F17C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1648>, <rasa.core.events.SlotSet object at 0x000001A3BF8F18C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1A48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8F1B08>, <rasa.core.events.Form object at 0x000001A3BF8F1C48>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1B88>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1CC8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1D08>, <rasa.core.events.Form object at 0x000001A3BF8F1D88>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1C08>]),
      '57_25447c1fd41449ad956c3feef44dd07b': StoryStep(block_name='equipInstall Generated Form Story 5', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8F1F88>, <rasa.core.events.SlotSet object at 0x000001A3BF8F1E88>, <rasa.core.events.SlotSet object at 0x000001A3BF8F50C8>, <rasa.core.events.ActionExecuted object at 0x000001A3BF8F5248>, <rasa.core.events.Form object at 0x000001A3BF8F5348>, <rasa.core.events.SlotSet object at 0x000001A3BF8F52C8>, <rasa.core.events.SlotSet object at 0x000001A3BF8F53C8>, <rasa.core.events.UserUttered object at 0x000001A3BF70A108>, <rasa.core.events.SlotSet object at 0x000001A3BF70AB08>, <rasa.core.events.ActionExecuted object at 0x000001A3BF70A608>, <rasa.core.events.SlotSet object at 0x000001A3BF733748>, <rasa.core.events.Form object at 0x000001A3BF8BC808>, <rasa.core.events.SlotSet object at 0x000001A3BF87D5C8>]),
      '58_94269af447734bf8a680276b6186e2cc': StoryStep(block_name='deviceParameter path_1', start_checkpoints=[Checkpoint(name='STORY_START', conditions={})], end_checkpoints=[], events=[<rasa.core.events.UserUttered object at 0x000001A3BF8F1E48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF87D508>, <rasa.core.events.Form object at 0x000001A3BF876E88>, <rasa.core.events.Form object at 0x000001A3BF876E48>, <rasa.core.events.ActionExecuted object at 0x000001A3BF6F1BC8>])},
     'ordered_ids': deque(['30_59cb32cb69a14425b5ec7ceb21dfab88',
            '31_5436161f1ba141d79006debc4e61a004',
            '32_6ab17da8458f44db8c4f0d809fb92cc6',
            '33_ddbbf2fcb04745428807e5f0591ff77c',
            '34_e543523258f14563926bc7934c17b936',
            '35_4b422fd5799b48d5b0d857e862111ec3',
            '36_92b0ebd96d034401bc6cf5e6631ee0aa',
            '37_32e443a972d34a6c93342ffa794248e0',
            '38_ff72b1e474eb4a85b76bf1a9cfa886e9',
            '39_4ce9f23606444e6e8be2cbceb62f4eec',
            '40_a171f49097f9400993a175980c61fc91',
            '41_0881e034b025458983ecbc7a875abd25',
            '42_2ac0edb631204c4fb0f5cc22da3f14ae',
            '43_acc7066f4e584ec487380350a14608bc',
            '44_554dd70f09f94f6aaa1bcd4122385a7a',
            '45_72ec892318b848f4b7eaf96768b5ab60',
            '46_a96f4cf20fd041be82c0fbec6f1422bc',
            '47_9ced12c5cb7d4d0aaff4753e795e0b5c',
            '48_6f113e0986604e38a6c45f013236a895',
            '49_9da19de845dd42c896098e07018d122a',
            '50_3bee6a16c39c4eabaa1278f9bd96cfc2',
            '51_20ad94aadf594d179b405c4903357db7',
            '52_3825abe773fc4daca3da45e135d04e4d',
            '53_37acd2b8d2a34cdba5e139047e57b358',
            '54_05045c417be746e89c91eac0af345766',
            '55_f6bc73cec1d44f4db2670a88fd2da682',
            '56_ecc468847db04cf58456cb7d6e916fa3',
            '57_25447c1fd41449ad956c3feef44dd07b',
            '58_94269af447734bf8a680276b6186e2cc']),
     'cyclic_edge_ids': [],
     'story_end_checkpoints': {}}



nlu.md被解析成rasa.nlu.training_data.message.Message格式数据


```python
nlu_data = await file_importer.get_nlu_data() #返回类 rasa.nlu.training_data.training_data.TrainingData
nlu_data.__dict__
```




    {'training_examples': [<rasa.nlu.training_data.message.Message at 0x1a3bf877a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8773c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8776c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf877c88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf877448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf877f48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf877148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf877248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf877fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf877288>,
      <rasa.nlu.training_data.message.Message at 0x1a3a4106d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bc340d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf877608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf686a88>,
      <rasa.nlu.training_data.message.Message at 0x1a3a4106d88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf686908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf6918c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf691c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf691dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf691708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf6866c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf691cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf691748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bc940d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bc940cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3a406cc48>,
      <rasa.nlu.training_data.message.Message at 0x1a3a401c3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3a401c688>,
      <rasa.nlu.training_data.message.Message at 0x1a3a406cd08>,
      <rasa.nlu.training_data.message.Message at 0x1a3a401c548>,
      <rasa.nlu.training_data.message.Message at 0x1a3a401c308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf6f1748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf720548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf6f1e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf6f19c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf70a488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf71c708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf72cb08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf87d908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf87de48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf87d448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf87d8c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf701108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852f48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852bc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852a88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf852e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8523c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf733a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf69da88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf73a808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7abd08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8c1d88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8c1388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8c1408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8c1348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf715388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf715548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf715088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf715b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf715148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf715188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8765c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf876748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8eebc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf92f288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf92f5c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf92f2c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf92fa48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf92f788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf92fdc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf92fb88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf92ff88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf935408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf935608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf935448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf935a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9357c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf935e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf935bc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf935fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8b0688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93d248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93d548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93d388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93d848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93d688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93db48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93d988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93de48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93dc88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93df88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93a2c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93a508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93a708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93a548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93aa08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93a848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93aa48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93ac48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf93a308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf947388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf947588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf947788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf947988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf947b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf947dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf947fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf948208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf948148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf948448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf948648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf948848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf948a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf948c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9590c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf959108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf959308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf959508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf959708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf959908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf959b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf959d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf966208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9665c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9669c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf966d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf966fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf96b1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf96b588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf96b908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf96bc88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf96bf08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf96bf48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf975608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf975908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf975c08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf975e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf975ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf97d3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf97d5c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf97d7c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf97d9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf97dbc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf97dec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf97df08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8b9688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8b9a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8b9548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8b9ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9841c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9846c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9849c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf984bc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf984e48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf984d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf95d308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf95d508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf95d708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf95da48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf95d908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf95dc08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9900c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf990108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf990308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf990508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf990708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf990908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf990b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf990d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf993208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf993148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf993748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf993948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf993b48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf993d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf993f48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf993f88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9a4448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9a46c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9a4a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9a4dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9a4c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ab348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ab848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9abb48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9abe48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9abd48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b1248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b1708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b1908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b1c08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b1e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b1d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b81c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b8648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b88c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b8bc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b8dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b8fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b6108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b6488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b6708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b6a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b6c08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b6e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9b6dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf99f188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf99f488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf99f688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf99f8c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf99fc48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf99fb48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9cd1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9cd688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9cd9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9cdb48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9cdd88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9cdd08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d4288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d4708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d4908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d4c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d4fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d8148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d8548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d87c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d89c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d8cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d8ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9d8e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c8348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c8208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c8b48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c8ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c8c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ea208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ea148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ea348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ea548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ea748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ea948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9eab08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9ead08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f72c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f7648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f7848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f7a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f7d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f7fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9fa288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9fa688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9fa9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9facc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9fac08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f1208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f1148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f1988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f1b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f1e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9f1cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa26088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa260c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa262c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa264c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa266c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa268c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa26ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa26cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa2c0c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa2c108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa2c308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa2c8c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa2cac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa2ccc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa2cec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa2ce08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa32308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa32508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa32708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa32908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa32b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa32d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa32f08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa32f48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa1f608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa1f808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa1f748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa1f948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa1fb88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa1fe88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa25f48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa43048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa431c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa43348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa434c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa43648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa437c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa43948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa43ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa43c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa43dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa59dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa5e248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa5e1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa5e448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa5e6c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa5e948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa5ebc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa67088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa670c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa67348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa675c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa67848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa67ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa67e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6b308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6b288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6b508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6b788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6ba08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6bdc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6f248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6f1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6f448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6f6c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6f948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa6fbc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa73048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa73108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa73388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa73608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa73888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa73b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa73ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa64408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa64688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa64888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa64ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa64d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa64cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa830c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa83188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa83488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa83788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa83a88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa88088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa880c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa883c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa886c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa889c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa88c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa8e208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa8e188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa8e408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa8e688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa8e908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa8eb88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa99088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa99148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa993c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa99648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa998c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa99b48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa9d048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa9d088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa9d388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa9d688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa9d988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa9de08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaa5388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaa52c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaa55c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaa58c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaa5d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaab2c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaab208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaab508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaab808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaabb08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaae048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaae108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaae388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaae608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaae888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaaeb08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaaeec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfab2048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfab22c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfab2548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfab27c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfab2a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfab2e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabb288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabb208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabb488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabb708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabb988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabbc08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac3f48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac8048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac81c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac8348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac84c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac8648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac87c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac8948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac8ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac8c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfac8dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bface188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bface108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bface288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bface408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bface588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bface708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bface888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfacea08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaceb88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaced08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfacef48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad5048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad51c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad5348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad54c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad5648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad57c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad5948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad5ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad5c48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfad5dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfae4d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaeb208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaeb188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaeb388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaeb588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaeb788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaeb988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaebb88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaebe88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf0248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf01c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf0388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf0508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf0708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf0908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf0b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf0d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaef1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaef148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaef348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaef548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaef748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaef948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaefb48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaefd48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf9208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf9188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf9388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf9588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf9788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf9988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf9b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfaf9e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3a248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3a188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3a388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3a588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3aa88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3ac08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3ad88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3af08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa3ae88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfafd048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfafd0c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfafd2c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfafd4c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfafd6c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfafd8c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfafdac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfafdf08>,
      <rasa.nlu.training_data.message.Message at 0x1a3a4134ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8a2e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd50b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd50bc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd50688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd50388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd50088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd4ad48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd4aa48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd4a748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd4a448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd4a148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd44e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd44b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd44808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd44448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfd44048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7a7648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7a76c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7a7488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa12108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa12488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa12708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa12988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa12b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa12dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa12d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa021c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa026c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa023c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa02788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa02a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa02e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9df388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9df608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9df888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9dfb08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9dfe88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9dfdc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf98a188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf98a508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf98a948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf98ac08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf98ae88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf98aec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf953048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf953088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf953288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf953488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf953688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf953888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf953cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf953988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf979388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf979208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9794c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf979748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9799c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf979e48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c2248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c2648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c28c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c2ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c2ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9c2f08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9e42c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9e4308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9e4588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9e4788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9e4ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9e4cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa09388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa093c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa096c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa09d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa0f088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa0f0c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa0f2c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa0f508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa0f788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa0fa88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa0fc48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa782c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa78448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa785c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa78748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa78a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa78bc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa78d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa78ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa78fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa78608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfada248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfada3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfada548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfada6c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfada848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfada9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfadab48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfadacc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfadae48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfadae88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb021c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb023c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb028c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb029c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02bc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb02fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa191c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa19488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa19708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa199c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa19c88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa19348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa19508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4e048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4e1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4e348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4e788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4e908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4ea88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4ec08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4ed88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4ef88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa4edc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa53248>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa533c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa53548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa536c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa53848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa539c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa53b48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa53cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa53f08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa53e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa7d208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa7d188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa7d308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa7d488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa7d608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa7d788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa7d9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa7de48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa93088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa930c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa93348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa935c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa93988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa93c08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfa93848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabe308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabe288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabe508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabe788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabeb48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfabed08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7a23c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7a28c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf857288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf857d88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf857ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7a8308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7a8788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf7a1648>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf69cec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf742688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf705fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb0c048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb0cc08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb0c708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb0c208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb0c3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb0c8c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb0cdc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb15c88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb15788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb15288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb15308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb15808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb15d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb11e48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb11948>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb11448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb11108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb11608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb11b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb10ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb109c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb104c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb10048>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb10548>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb10a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb10f48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb19b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb19688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb19188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb19348>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb19848>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb19d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb1dd88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb1d788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb1d208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb1d3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb1da48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8f8fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8f89c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8f83c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8f8208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8f8808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8f8e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8fec08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8fe608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8fe148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8fe748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf8fed48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf904cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9046c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9040c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf904508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf904b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf90cf08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf90c908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf90c308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf90c448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf90ca48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf913fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9139c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf9133c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf913208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf913808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bf913e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb24c08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb24608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb24148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb24748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb24d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb2acc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb2a6c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb2a0c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb2a508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb2ab08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb30f08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb30908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb30308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb30448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb30a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb34fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb349c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb343c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb34208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb34808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb34e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3ac08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3a608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3a148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3a748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3ad48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3fcc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3f6c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3f0c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3f508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb3fb08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb46f08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb46908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb46308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb46448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb46a48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb4afc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb4a9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb4a3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb4a208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb4a808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb4ae08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb50c08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb50608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb50148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb50748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb50d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb55cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb556c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb550c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb55508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb55b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb5af48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb5a908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb5a308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb5a448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb5aa48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb60e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb60888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb60288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb603c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb609c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb60fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb66ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb664c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb66188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb66788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb66d88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb6cb88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb6c588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb6c0c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb6c6c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb6ccc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb72dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb727c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb721c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb72488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb72a88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb79e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb79888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb79288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb793c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb799c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb79fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb7fac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb7f4c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb7f188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb7f788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb7fd88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb85b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb85588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb850c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb856c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb85cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8adc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8a7c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8a1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8a488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8aa88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8fe88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8f888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8f288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8f3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8f9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb8ffc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb95ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb954c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb95188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb95788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb95d88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9bb88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9b588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9b0c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9b6c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9bcc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9fdc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9f7c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9f1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9f488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfb9fa88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfba6e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfba6888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfba6288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfba63c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfba69c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfba6fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbadac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbad4c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbad188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbad788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbadd88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb3b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb3588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb30c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb36c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb3cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb8dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb87c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb81c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb8488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbb8a88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbbee88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbbe888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbbe288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbbe3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbbe9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbbefc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc4ac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc44c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc4188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc4788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc4d88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc9b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc9588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc90c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc96c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbc9cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd0dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd07c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd01c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd0488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd0a88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd6e88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd6888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd6288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd63c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd69c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbd6fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbddac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbdd4c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbdd188>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbdd788>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbddd88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe2b88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe2588>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe20c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe26c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe2cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe7dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe77c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe71c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe7488>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbe7a88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbedec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbed888>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbed288>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbed3c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbed9c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf2f88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf2988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf2388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf21c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf27c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf2dc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf8bc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf85c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf8108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf8708>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbf8d08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbfcc88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbfc688>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbfc088>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbfc4c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfbfcac8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc05ec8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc058c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc052c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc05408>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc05a08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc0bf88>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc0b988>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc0b388>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc0b1c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc0b7c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc0bdc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc10c08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc10608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc10108>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc10748>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc10d48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc16cc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc166c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc160c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc16508>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc16b08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc1cf08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc1c908>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc1c308>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc1c448>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc1ca48>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc21fc8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc219c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc213c8>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc21208>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc21808>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc21e08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc27c08>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc27608>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc27148>,
      <rasa.nlu.training_data.message.Message at 0x1a3bfc27748>,
      ...],
     'entity_synonyms': {},
     'regex_features': [],
     'lookup_tables': [],
     'nlg_stories': {}}



rasa.nlu.training_data.message.Message格式数据


```python
nlu_data.__dict__['training_examples'][910].__dict__
```




    {'text': 'LED1闪烁',
     'time': None,
     'data': {'intent': 'inform',
      'entities': [{'start': 0,
        'end': 6,
        'value': 'LED1闪烁',
        'entity': 'FaultPhenomenon'}]},
     'output_properties': set()}



config数据


```python
config = await file_importer.get_config() #返回数据 dict
NLU_config = nlu_config.load(configs_file) #见文件metadata_parsing.txt
Core_config = core_config.load(configs_file) #见文件metadata_parsing.txt
NLU_config.as_dict(),Core_config
```

    D:\ProgramFiles\Anaconda3\lib\site-packages\rasa\core\policies\ensemble.py:318: FutureWarning: 'KerasPolicy' is deprecated and will be removed in version 2.0. Use 'TEDPolicy' instead.
      policy_object = constr_func(**policy)
    




    ({'language': 'zh',
      'pipeline': [{'name': 'JiebaTokenizer'},
       {'name': 'KashgariEntityExtractor',
        'bert_model_path': 'chinese_L-12_H-768_A-12'},
       {'name': 'RegexExtractor', 'regex_ner_entity': ['EqpType']},
       {'name': 'KashgariIntentClassifier',
        'bert_model_path': 'chinese_L-12_H-768_A-12'}],
      'data': None,
      'policies': [{'name': 'TwoStageFallbackPolicy',
        'core_threshold': 0.1,
        'nlu_threshold': 0.1,
        'fallback_core_action_name': 'action_default_fallback',
        'fallback_nlu_action_name': 'action_handoff_to_human',
        'deny_suggestion_intent_name': 'out_of_scope'},
       {'name': 'MemoizationPolicy', 'max_history': 3},
       {'name': 'FormPolicy'},
       {'name': 'MappingPolicy'},
       {'name': 'KerasPolicy',
        'featurizer': [{'name': 'MaxHistoryTrackerFeaturizer',
          'max_history': 5,
          'state_featurizer': [{'name': 'BinarySingleStateFeaturizer'}]}]}]},
     [<rasa.core.policies.two_stage_fallback.TwoStageFallbackPolicy at 0x1a3bf87dd08>,
      <rasa.core.policies.memoization.MemoizationPolicy at 0x1a3bf852988>,
      <rasa.core.policies.form_policy.FormPolicy at 0x1a3bfd10048>,
      <rasa.core.policies.mapping_policy.MappingPolicy at 0x1a3ddf1bd88>,
      <rasa.core.policies.keras_policy.KerasPolicy at 0x1a3ddf1be88>])



domain_states 为对话系统domain中包含的对话状态编码，用于生成对话状态向量（作为Policy的输入）


```python
domain_dict = domain.as_dict()
domain_states = domain.input_state_map

domain_dict, domain_states
```




    ({'config': {'store_entities_as_slots': True},
      'session_config': {'session_expiration_time': 0,
       'carry_over_slots_to_new_session': True},
      'intents': [{'faultDiag': {'triggers': 'utter_faultDiag',
         'use_entities': True}},
       {'sysFaultDiag': {'use_entities': True}},
       {'equipFaultDiag': {'use_entities': True}},
       {'maintenance': {'triggers': 'utter_maintenance', 'use_entities': True}},
       {'sysMaintenance': {'use_entities': True}},
       {'equipMaintenance': {'use_entities': True}},
       {'operation': {'triggers': 'utter_operation', 'use_entities': True}},
       {'sysOperation': {'use_entities': True}},
       {'equipOperation': {'use_entities': True}},
       {'install': {'triggers': 'utter_install', 'use_entities': True}},
       {'equipInstall': {'use_entities': True}},
       {'sysInstall': {'use_entities': True}},
       {'deviceParameter': {'use_entities': True}},
       {'inform': {'use_entities': True}},
       {'greet': {'triggers': 'utter_greet', 'use_entities': True}},
       {'goodbye': {'triggers': 'utter_goodbye', 'use_entities': True}},
       {'affirm': {'use_entities': True}},
       {'deny': {'use_entities': True}},
       {'thankyou': {'use_entities': True}},
       {'chitchat': {'triggers': 'action_tuling_api', 'use_entities': True}},
       {'stop': {'triggers': 'utter_stop', 'use_entities': True}},
       {'continue': {'triggers': 'action_continue', 'use_entities': True}},
       {'out': {'triggers': 'action_out', 'use_entities': True}},
       {'askBotFunction': {'triggers': 'action_bot_function',
         'use_entities': True}},
       {'askIsBot': {'triggers': 'action_is_bot', 'use_entities': True}},
       {'askPoiet': {'triggers': 'utter_askPoiet', 'use_entities': True}},
       {'insult': {'triggers': 'action_respond_insult', 'use_entities': True}},
       {'opinion+positive': {'triggers': 'utter_positive_feedback_reaction',
         'use_entities': True}},
       {'opinion+negative': {'triggers': 'utter_imporvement_request',
         'use_entities': True}},
       {'out_of_scope': {'triggers': 'utter_ask_rephrase', 'use_entities': True}},
       {'unknow': {'use_entities': True}}],
      'entities': ['Brand',
       'EqpName',
       'EqpType',
       'FaultPhenomenon',
       'FtCode',
       'InstallApproach',
       'Installation',
       'Maintaining',
       'MajorName',
       'OpApproach',
       'OpManagement',
       'OpRegulation',
       'OutMark',
       'Param',
       'RunSystem',
       'Setting',
       'SysName',
       'TestRun',
       'parameter'],
      'slots': {'Brand': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'EqpName': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'EqpType': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'FaultPhenomenon': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'FtCode': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'InstallApproach': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'Maintaining': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'OpApproach': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'OutMark': {'type': 'rasa.core.slots.BooleanSlot',
        'initial_value': None,
        'auto_fill': True},
       'Param': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'SysName': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True},
       'feedback': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': False},
       'requested_slot': {'type': 'rasa.core.slots.UnfeaturizedSlot',
        'initial_value': None,
        'auto_fill': True}},
      'responses': {'utter_faultDiag': [{'text': '您要查询系统故障，还是设备故障？',
         'buttons': [{'title': '系统故障:(例如，中央空调供水温度降不下来)',
           'payload': '/sysFaultDiag'},
          {'title': '设备故障:(例如，水泵喘振)', 'payload': '/equipFaultDiag'}]}],
       'utter_maintenance': [{'text': '您要查询系统维保，还是设备维保？',
         'buttons': [{'title': '系统维保方法:(例如，中央空调冷却水水质保养方法)',
           'payload': '/sysMaintenance'},
          {'title': '设备维保方法:(例如，冷水机组冷凝器通炮方法)', 'payload': '/equipMaintenance'}]}],
       'utter_operation': [{'text': '您想查询系统操作方法，还是设备操作方法？',
         'buttons': [{'title': '系统操作方法:(例如，集中式中央空调制冷站启动方法)',
           'payload': '/sysOperation'},
          {'title': '设备操作方法:(例如，冷水机组的启动顺序)', 'payload': '/equipOperation'}]}],
       'utter_install': [{'text': '您想查询系统安装（或施工）方法，还是设备安装方法？',
         'buttons': [{'title': '系统安装方法:(例如，中央空调制冷站施工方法)',
           'payload': '/sysInstall'},
          {'title': '设备安装方法:(例如，冷水机组的安装方法)', 'payload': '/equipInstall'}]}],
       'utter_inquireEquipment': [{'text': '什么设备名称?(冷水机组/水泵/风机盘管/...)'}],
       'utter_inquireBrand': [{'text': '什么品牌?'}],
       'utter_inquireType:': [{'text': '什么型号？'}],
       'utter_inquireEquipmentFualt': [{'text': '什么故障代码？'}],
       'utter_inquireSystem': [{'text': '什么系统？(给排水系统/供配电系统/空调系统/通风系统/消防系统/...)'}],
       'utter_inquireSystemFualt': [{'text': '什么故障现象？'}],
       'utter_inquireFeedback': [{'text': '是否帮到您了？'}],
       'utter_submit': [{'text': '提交'}],
       'utter_no_equipment': [{'text': '该设备没有找到！'}],
       'utter_no_brand': [{'text': '该品牌没有找到！'}],
       'utter_no_type': [{'text': '该型号没有找到，请核对！'}],
       'utter_no_equipmentFualt': [{'text': '该故障代码没有找到'}],
       'utter_no_system': [{'text': '该系统名称没有找到'}],
       'utter_no_systemFualt': [{'text': '没有找到匹配的故障现象'}],
       'utter_ask_EqpName': [{'text': '请输入设备名称'}, {'text': '设备名称是什么？'}],
       'utter_ask_Brand': [{'text': '请输入设备品牌'}, {'text': '是哪个设备品牌？'}],
       'utter_ask_EqpType': [{'text': '请输入设备型号：'}, {'text': '是哪种设备型号？'}],
       'utter_ask_FtCode': [{'text': '请输入故障代码：'}, {'text': '有什么故障代码？'}],
       'utter_ask_SysName': [{'text': '请输入系统名称：'}, {'text': '系统名称是什么？'}],
       'utter_ask_FaultPhenomenon': [{'text': '请输入故障现象：'}, {'text': '有什么故障现象？'}],
       'utter_wait': [{'text': '正在查询'}],
       'utter_iamabot': [{'text': '我是一个智能问答机器人'}],
       'utter_function': [{'text': '我可以解答建筑运维中遇到的技术问题：如故障诊断、运行方法、维保方法、设备参数查询等等'}],
       'utter_default': [{'text': '听不懂'}, {'text': '能否换个说法？'}, {'text': '臣妾不懂哦'}],
       'utter_askPoiet': [{'text': '锄禾日当午，汗滴禾下土'},
        {'text': '鹅鹅鹅~'},
        {'text': '额，让我想想~'}],
       'utter_noworries': [{'text': '不用谢'},
        {'text': '应该的'},
        {'text': '没事，不客气 :)'}],
       'utter_chitchat': [{'text': '抱歉，现在很忙，没功夫闲聊'}],
       'utter_ask_continue': [{'text': '继续?'},
        {'text': '是否继续?'},
        {'text': '继续对话?'}],
       'utter_respond_insult': [{'text': '笑一笑十年少，何必那么生气呢'}, {'text': '微笑使人进步'}],
       'utter_greet': [{'text': '你好~我是智能建筑运维助手，有运维问题请随时找我~'},
        {'text': '您好！关于空调、消防、照明的运维问题可以随时咨询我哦！'},
        {'text': 'hi，我是iBot，您有什么问题？'},
        {'text': '你好，您有什么问题？'}],
       'utter_goodbye': [{'text': '再见~'},
        {'text': '拜拜~'},
        {'text': 'bye~'},
        {'text': '下次再见~'}],
       'utter_positive_fallback': [{'text': '你也好'}],
       'utter_ask_rephrase': [{'text': '没有理解您的意思，请重新输入'}],
       'utter_stop': [{'text': '暂停中',
         'buttons': [{'title': '退出', 'payload': '/out'},
          {'title': '继续', 'payload': '/continue'}]}],
       'utter_positive_feedback_reaction': [{'text': '你也好'}],
       'utter_imporvement_request': [{'text': '有没有改进的建议呢？'}],
       'utter_slots_equipFaultDiag': [{'text': '您提供的信息如下:\n - 设备名称: {EqpName}:\n - 设备品牌: {Brand}:\n - 设备型号: {EqpType}:\n - 故障代码: {FtCode}:\n - 故障现象: {FaultPhenomenon}:\n'}],
       'utter_slots_sysFaultDiag': [{'text': '您提供的系统故障诊断信息如下:\n - 系统名称: {SysName}:\n - 故障现象: {FaultPhenomenon}:\n'}],
       'utter_slots_equipMaintenance': [{'text': '您提供的信息如下:\n - 设备名称: {EqpName}:\n - 设备品牌: {Brand}:\n - 设备型号: {EqpType}:\n - 维保方法: {Maintaining}:\n'}],
       'utter_slots_sysMaintenance': [{'text': '您提供的系统维保信息如下:\n - 系统名称: {SysName}:\n - 维保方法: {Maintaining}:\n'}],
       'utter_slots_equipOperation': [{'text': '您提供的信息如下:\n - 设备名称: {EqpName}:\n - 设备品牌: {Brand}:\n - 设备型号: {EqpType}:\n - 操作方法: {OpApproach}:\n'}],
       'utter_slots_sysOperation': [{'text': '您提供的系统操作信息如下:\n - 系统名称: {SysName}:\n - 操作方法: {OpApproach}:\n'}],
       'utter_slots_equipInstall': [{'text': '您提供的信息如下:\n - 设备名称: {EqpName}:\n - 设备品牌: {Brand}:\n - 设备型号: {EqpType}:\n - 安装方法: {InstallApproach}:\n'}],
       'utter_slots_sysInstall': [{'text': '您提供的系统操作信息如下:\n - 系统名称: {SysName}:\n - 安装方法: {InstallApproach}:\n'}],
       'utter_slots_deviceParameter': [{'text': '您提供的信息如下:\n - 设备名称: {EqpName}:\n - 设备品牌: {Brand}:\n - 设备型号: {EqpType}:\n - 查询参数: {Param}:\n'}]},
      'actions': ['action_bot_function',
       'action_continue',
       'action_handoff_to_human',
       'action_is_bot',
       'action_out',
       'action_respond_insult',
       'action_tuling_api',
       'utter_askPoiet',
       'utter_ask_Brand',
       'utter_ask_EqpName',
       'utter_ask_EqpType',
       'utter_ask_FaultPhenomenon',
       'utter_ask_FtCode',
       'utter_ask_SysName',
       'utter_ask_continue',
       'utter_ask_rephrase',
       'utter_chitchat',
       'utter_default',
       'utter_faultDiag',
       'utter_function',
       'utter_goodbye',
       'utter_greet',
       'utter_iamabot',
       'utter_imporvement_request',
       'utter_inquireBrand',
       'utter_inquireEquipment',
       'utter_inquireEquipmentFualt',
       'utter_inquireFeedback',
       'utter_inquireSystem',
       'utter_inquireSystemFualt',
       'utter_inquireType:',
       'utter_install',
       'utter_maintenance',
       'utter_no_brand',
       'utter_no_equipment',
       'utter_no_equipmentFualt',
       'utter_no_system',
       'utter_no_systemFualt',
       'utter_no_type',
       'utter_noworries',
       'utter_operation',
       'utter_positive_fallback',
       'utter_positive_feedback_reaction',
       'utter_respond_insult',
       'utter_slots_deviceParameter',
       'utter_slots_equipFaultDiag',
       'utter_slots_equipInstall',
       'utter_slots_equipMaintenance',
       'utter_slots_equipOperation',
       'utter_slots_sysFaultDiag',
       'utter_slots_sysInstall',
       'utter_slots_sysMaintenance',
       'utter_slots_sysOperation',
       'utter_stop',
       'utter_submit',
       'utter_wait'],
      'forms': ['deviceParameter_form',
       'equipFaultDiag_form',
       'equipInstall_form',
       'equipMaintenance_form',
       'equipOperation_form',
       'sysFaultDiag_form',
       'sysInstall_form',
       'sysMaintenance_form',
       'sysOperation_form']},
     {'intent_affirm': 0,
      'intent_askBotFunction': 1,
      'intent_askIsBot': 2,
      'intent_askPoiet': 3,
      'intent_chitchat': 4,
      'intent_continue': 5,
      'intent_deny': 6,
      'intent_deviceParameter': 7,
      'intent_equipFaultDiag': 8,
      'intent_equipInstall': 9,
      'intent_equipMaintenance': 10,
      'intent_equipOperation': 11,
      'intent_faultDiag': 12,
      'intent_goodbye': 13,
      'intent_greet': 14,
      'intent_inform': 15,
      'intent_install': 16,
      'intent_insult': 17,
      'intent_maintenance': 18,
      'intent_operation': 19,
      'intent_opinion+negative': 20,
      'intent_opinion+positive': 21,
      'intent_out': 22,
      'intent_out_of_scope': 23,
      'intent_stop': 24,
      'intent_sysFaultDiag': 25,
      'intent_sysInstall': 26,
      'intent_sysMaintenance': 27,
      'intent_sysOperation': 28,
      'intent_thankyou': 29,
      'intent_unknow': 30,
      'entity_Brand': 31,
      'entity_EqpName': 32,
      'entity_EqpType': 33,
      'entity_FaultPhenomenon': 34,
      'entity_FtCode': 35,
      'entity_InstallApproach': 36,
      'entity_Installation': 37,
      'entity_Maintaining': 38,
      'entity_MajorName': 39,
      'entity_OpApproach': 40,
      'entity_OpManagement': 41,
      'entity_OpRegulation': 42,
      'entity_OutMark': 43,
      'entity_Param': 44,
      'entity_RunSystem': 45,
      'entity_Setting': 46,
      'entity_SysName': 47,
      'entity_TestRun': 48,
      'entity_parameter': 49,
      'slot_OutMark_0': 50,
      'slot_OutMark_1': 51,
      'prev_action_listen': 52,
      'prev_action_restart': 53,
      'prev_action_session_start': 54,
      'prev_action_default_fallback': 55,
      'prev_action_deactivate_form': 56,
      'prev_action_revert_fallback_events': 57,
      'prev_action_default_ask_affirmation': 58,
      'prev_action_default_ask_rephrase': 59,
      'prev_action_back': 60,
      'prev_action_bot_function': 61,
      'prev_action_continue': 62,
      'prev_action_handoff_to_human': 63,
      'prev_action_is_bot': 64,
      'prev_action_out': 65,
      'prev_action_respond_insult': 66,
      'prev_action_tuling_api': 67,
      'prev_utter_askPoiet': 68,
      'prev_utter_ask_Brand': 69,
      'prev_utter_ask_EqpName': 70,
      'prev_utter_ask_EqpType': 71,
      'prev_utter_ask_FaultPhenomenon': 72,
      'prev_utter_ask_FtCode': 73,
      'prev_utter_ask_SysName': 74,
      'prev_utter_ask_continue': 75,
      'prev_utter_ask_rephrase': 76,
      'prev_utter_chitchat': 77,
      'prev_utter_default': 78,
      'prev_utter_faultDiag': 79,
      'prev_utter_function': 80,
      'prev_utter_goodbye': 81,
      'prev_utter_greet': 82,
      'prev_utter_iamabot': 83,
      'prev_utter_imporvement_request': 84,
      'prev_utter_inquireBrand': 85,
      'prev_utter_inquireEquipment': 86,
      'prev_utter_inquireEquipmentFualt': 87,
      'prev_utter_inquireFeedback': 88,
      'prev_utter_inquireSystem': 89,
      'prev_utter_inquireSystemFualt': 90,
      'prev_utter_inquireType:': 91,
      'prev_utter_install': 92,
      'prev_utter_maintenance': 93,
      'prev_utter_no_brand': 94,
      'prev_utter_no_equipment': 95,
      'prev_utter_no_equipmentFualt': 96,
      'prev_utter_no_system': 97,
      'prev_utter_no_systemFualt': 98,
      'prev_utter_no_type': 99,
      'prev_utter_noworries': 100,
      'prev_utter_operation': 101,
      'prev_utter_positive_fallback': 102,
      'prev_utter_positive_feedback_reaction': 103,
      'prev_utter_respond_insult': 104,
      'prev_utter_slots_deviceParameter': 105,
      'prev_utter_slots_equipFaultDiag': 106,
      'prev_utter_slots_equipInstall': 107,
      'prev_utter_slots_equipMaintenance': 108,
      'prev_utter_slots_equipOperation': 109,
      'prev_utter_slots_sysFaultDiag': 110,
      'prev_utter_slots_sysInstall': 111,
      'prev_utter_slots_sysMaintenance': 112,
      'prev_utter_slots_sysOperation': 113,
      'prev_utter_stop': 114,
      'prev_utter_submit': 115,
      'prev_utter_wait': 116,
      'prev_deviceParameter_form': 117,
      'prev_equipFaultDiag_form': 118,
      'prev_equipInstall_form': 119,
      'prev_equipMaintenance_form': 120,
      'prev_equipOperation_form': 121,
      'prev_sysFaultDiag_form': 122,
      'prev_sysInstall_form': 123,
      'prev_sysMaintenance_form': 124,
      'prev_sysOperation_form': 125,
      'active_form_deviceParameter_form': 126,
      'active_form_equipFaultDiag_form': 127,
      'active_form_equipInstall_form': 128,
      'active_form_equipMaintenance_form': 129,
      'active_form_equipOperation_form': 130,
      'active_form_sysFaultDiag_form': 131,
      'active_form_sysInstall_form': 132,
      'active_form_sysMaintenance_form': 133,
      'active_form_sysOperation_form': 134})



## 3、模型训练方法解析

### 3.1 联合训练 nlu  +  core 

训练方式一


```python
train(domain_file, configs_file, training_files, output_path)
```

训练方式二


```python
await train_async(domain_file, configs_file, training_files, output_path)
```

### 3.2 独立训练nlu或core

单独训练 nlu


```python
await _train_nlu_with_validated_data(file_importer,
                                    output=output_path)
```

单独训练 core


```python
#详见9.3节
```

## 4、重要的类Agent解析

Agent类为最重要的Rasa功能提供了方便的接口。这包括训练、处理消息、加载对话模型、获取下一个动作和处理通道。

### 4.1 Agent 加载对话模型


```python
agent = Agent()
model_path = r'models/models_15/20200420-120239.tar.gz'
agent = agent.load(model_path) 
#loaded_agent = await load_agent() #agent.load的底层函数

agent.__dict__
```

    obj: None
    obj: C:\Users\86185\AppData\Local\Temp\tmpzu1kcsw_\nlu
    

    WARNING:root:Sequence length will auto set at 95% of sequence length
    WARNING:root:Sequence length will auto set at 95% of sequence length
    WARNING:root:Model will be built until sequence length is determined
    WARNING:root:Sequence length will auto set at 95% of sequence length
    WARNING:root:Sequence length will auto set at 95% of sequence length
    WARNING:root:Model will be built until sequence length is determined
    D:\ProgramFiles\Anaconda3\lib\site-packages\rasa\core\policies\keras_policy.py:265: FutureWarning: 'KerasPolicy' is deprecated and will be removed in version 2.0. Use 'TEDPolicy' instead.
      current_epoch=meta["epochs"],
    

    obj: <rasa.core.interpreter.RasaNLUInterpreter object at 0x0000025FA7775148>
    




    {'domain': <rasa.core.domain.Domain at 0x26009d8dd08>,
     'policy_ensemble': <rasa.core.policies.ensemble.SimplePolicyEnsemble at 0x25fddc58d88>,
     'interpreter': <rasa.core.interpreter.RasaNLUInterpreter at 0x25fa7775148>,
     'nlg': <rasa.core.nlg.template.TemplatedNaturalLanguageGenerator at 0x26004c06f88>,
     'tracker_store': <rasa.core.tracker_store.FailSafeTrackerStore at 0x26009d9cd88>,
     'lock_store': <rasa.core.lock_store.InMemoryLockStore at 0x26009d9c488>,
     'action_endpoint': None,
     'fingerprint': '9ce015e0043d488fb7f33dff014acd14',
     'model_directory': 'C:\\Users\\86185\\AppData\\Local\\Temp\\tmpzu1kcsw_',
     'model_server': None,
     'remote_storage': None,
     'path_to_model_archive': None}



### 4.2 Agent 调取nlu模型解析用户消息

解析出用户消息中的意图和实体


```python
ms_text = '我的空调有问题'
await agent.parse_message_using_nlu_interpreter(ms_text) 
```

    Building prefix dict from the default dictionary ...
    DEBUG:jieba:Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\86185\AppData\Local\Temp\jieba.cache
    DEBUG:jieba:Loading model from cache C:\Users\86185\AppData\Local\Temp\jieba.cache
    Loading model cost 0.621 seconds.
    DEBUG:jieba:Loading model cost 0.621 seconds.
    Prefix dict has been built succesfully.
    DEBUG:jieba:Prefix dict has been built succesfully.
    




    {'intent': {'name': 'faultDiag', 'confidence': 0.8657001852989197},
     'entities': [{'start': 2,
       'end': 4,
       'value': '空调',
       'entity': 'MajorName',
       'extractor': 'RegexExtractor'}],
     'intent_ranking': [{'name': 'faultDiag', 'confidence': 0.8657001852989197},
      {'name': 'equipFaultDiag', 'confidence': 0.04571297764778137},
      {'name': 'inform', 'confidence': 0.028037212789058685},
      {'name': 'opinion+negtive', 'confidence': 0.010507158003747463},
      {'name': 'unknow', 'confidence': 0.006765459664165974},
      {'name': 'greet', 'confidence': 0.006672943476587534},
      {'name': 'maintenance', 'confidence': 0.004478101618587971},
      {'name': 'sysFaultDiag', 'confidence': 0.0033905007876455784},
      {'name': 'thankyou', 'confidence': 0.003203833010047674},
      {'name': 'insult', 'confidence': 0.0031317586544901133}],
     'text': '我的空调有问题'}



### 4.3 、Agent 接收用户消息调取对话管理模块并产生系统输出

agent接收并处理用户消息，然后trackerstore得到当前对话状态，输入policy ensamble，得到agent action输出


```python
ms_text = '我的空调有问题'
#ms_text = '唱一首歌吧'
uM = UserMessage(ms_text) #用户消息文本封装
print(uM.__dict__)
#agent加载外部Action服务端口
agent.action_endpoint = read_endpoint_config('configs/endpoints.yml',endpoint_type='action_endpoint')
#系统action输出
await agent.handle_message(uM)
```

    {'text': '我的空调有问题', 'message_id': 'c0ef6e9a2fa0483fa974e7bdcf14a65e', 'output_channel': <rasa.core.channels.channel.CollectingOutputChannel object at 0x0000025FE70070C8>, 'sender_id': 'default', 'input_channel': None, 'parse_data': None, 'metadata': None}
    




    [{'recipient_id': 'default',
      'text': '您要查询系统故障，还是设备故障？',
      'buttons': [{'payload': '/sysFaultDiag', 'title': '系统故障:(例如，中央空调供水温度降不下来)'},
       {'payload': '/equipFaultDiag', 'title': '设备故障:(例如，水泵喘振)'}]}]



agent基于用户sender_id的当前对话状态，输出action的预测分值、最大分值对应策略、tracker_current_state


```python
predict_next = await agent.predict_next(sender_id='default',output_channel=CollectingOutputChannel) #只输出动作，没有生成回复
predict_next
```




    {'scores': [{'action': 'action_listen', 'score': 0.0},
      {'action': 'action_restart', 'score': 0.0},
      {'action': 'action_session_start', 'score': 0.0},
      {'action': 'action_default_fallback', 'score': 0.0},
      {'action': 'action_deactivate_form', 'score': 0.0},
      {'action': 'action_revert_fallback_events', 'score': 0.0},
      {'action': 'action_default_ask_affirmation', 'score': 0.0},
      {'action': 'action_default_ask_rephrase', 'score': 0.0},
      {'action': 'action_back', 'score': 0.0},
      {'action': 'action_bot_function', 'score': 0.0},
      {'action': 'action_continue', 'score': 0.0},
      {'action': 'action_handoff_to_human', 'score': 0.0},
      {'action': 'action_is_bot', 'score': 0.0},
      {'action': 'action_out', 'score': 0.0},
      {'action': 'action_respond_insult', 'score': 0.0},
      {'action': 'action_tuling_api', 'score': 0.0},
      {'action': 'utter_askPoiet', 'score': 0.0},
      {'action': 'utter_ask_Brand', 'score': 0.0},
      {'action': 'utter_ask_EqpName', 'score': 0.0},
      {'action': 'utter_ask_EqpType', 'score': 0.0},
      {'action': 'utter_ask_FaultPhenomenon', 'score': 0.0},
      {'action': 'utter_ask_FtCode', 'score': 0.0},
      {'action': 'utter_ask_SysName', 'score': 0.0},
      {'action': 'utter_ask_continue', 'score': 0.0},
      {'action': 'utter_ask_rephrase', 'score': 0.0},
      {'action': 'utter_chitchat', 'score': 0.0},
      {'action': 'utter_default', 'score': 0.0},
      {'action': 'utter_faultDiag', 'score': 1},
      {'action': 'utter_function', 'score': 0.0},
      {'action': 'utter_goodbye', 'score': 0.0},
      {'action': 'utter_greet', 'score': 0.0},
      {'action': 'utter_iamabot', 'score': 0.0},
      {'action': 'utter_imporvement_request', 'score': 0.0},
      {'action': 'utter_inquireBrand', 'score': 0.0},
      {'action': 'utter_inquireEquipment', 'score': 0.0},
      {'action': 'utter_inquireEquipmentFualt', 'score': 0.0},
      {'action': 'utter_inquireFeedback', 'score': 0.0},
      {'action': 'utter_inquireSystem', 'score': 0.0},
      {'action': 'utter_inquireSystemFualt', 'score': 0.0},
      {'action': 'utter_inquireType:', 'score': 0.0},
      {'action': 'utter_install', 'score': 0.0},
      {'action': 'utter_maintenance', 'score': 0.0},
      {'action': 'utter_no_brand', 'score': 0.0},
      {'action': 'utter_no_equipment', 'score': 0.0},
      {'action': 'utter_no_equipmentFualt', 'score': 0.0},
      {'action': 'utter_no_system', 'score': 0.0},
      {'action': 'utter_no_systemFualt', 'score': 0.0},
      {'action': 'utter_no_type', 'score': 0.0},
      {'action': 'utter_noworries', 'score': 0.0},
      {'action': 'utter_operation', 'score': 0.0},
      {'action': 'utter_positive_fallback', 'score': 0.0},
      {'action': 'utter_positive_feedback_reaction', 'score': 0.0},
      {'action': 'utter_respond_insult', 'score': 0.0},
      {'action': 'utter_slots_deviceParameter', 'score': 0.0},
      {'action': 'utter_slots_equipFaultDiag', 'score': 0.0},
      {'action': 'utter_slots_equipInstall', 'score': 0.0},
      {'action': 'utter_slots_equipMaintenance', 'score': 0.0},
      {'action': 'utter_slots_equipOperation', 'score': 0.0},
      {'action': 'utter_slots_sysFaultDiag', 'score': 0.0},
      {'action': 'utter_slots_sysInstall', 'score': 0.0},
      {'action': 'utter_slots_sysMaintenance', 'score': 0.0},
      {'action': 'utter_slots_sysOperation', 'score': 0.0},
      {'action': 'utter_stop', 'score': 0.0},
      {'action': 'utter_submit', 'score': 0.0},
      {'action': 'utter_wait', 'score': 0.0},
      {'action': 'deviceParameter_form', 'score': 0.0},
      {'action': 'equipFaultDiag_form', 'score': 0.0},
      {'action': 'equipInstall_form', 'score': 0.0},
      {'action': 'equipMaintenance_form', 'score': 0.0},
      {'action': 'equipOperation_form', 'score': 0.0},
      {'action': 'sysFaultDiag_form', 'score': 0.0},
      {'action': 'sysInstall_form', 'score': 0.0},
      {'action': 'sysMaintenance_form', 'score': 0.0},
      {'action': 'sysOperation_form', 'score': 0.0}],
     'policy': 'policy_3_MappingPolicy',
     'confidence': 1.0,
     'tracker': {'sender_id': 'default',
      'slots': {'Brand': None,
       'EqpName': None,
       'EqpType': None,
       'FaultPhenomenon': None,
       'FtCode': None,
       'InstallApproach': None,
       'Maintaining': None,
       'OpApproach': None,
       'OutMark': None,
       'Param': None,
       'SysName': None,
       'feedback': None,
       'requested_slot': None},
      'latest_message': {'intent': {'name': 'faultDiag',
        'confidence': 0.8657001852989197},
       'entities': [{'start': 2,
         'end': 4,
         'value': '空调',
         'entity': 'MajorName',
         'extractor': 'RegexExtractor'}],
       'intent_ranking': [{'name': 'faultDiag', 'confidence': 0.8657001852989197},
        {'name': 'equipFaultDiag', 'confidence': 0.04571297764778137},
        {'name': 'inform', 'confidence': 0.028037212789058685},
        {'name': 'opinion+negtive', 'confidence': 0.010507158003747463},
        {'name': 'unknow', 'confidence': 0.006765459664165974},
        {'name': 'greet', 'confidence': 0.006672943476587534},
        {'name': 'maintenance', 'confidence': 0.004478101618587971},
        {'name': 'sysFaultDiag', 'confidence': 0.0033905007876455784},
        {'name': 'thankyou', 'confidence': 0.003203833010047674},
        {'name': 'insult', 'confidence': 0.0031317586544901133}],
       'text': '我的空调有问题'},
      'latest_event_time': 1589357962.3806748,
      'followup_action': None,
      'paused': False,
      'events': [{'event': 'action',
        'timestamp': 1589357942.8721497,
        'name': 'action_session_start',
        'policy': None,
        'confidence': None},
       {'event': 'session_started', 'timestamp': 1589357942.8721497},
       {'event': 'action',
        'timestamp': 1589357942.8721497,
        'name': 'action_listen',
        'policy': None,
        'confidence': None},
       {'event': 'user',
        'timestamp': 1589357943.052667,
        'text': '我的空调有问题',
        'parse_data': {'intent': {'name': 'faultDiag',
          'confidence': 0.8657001852989197},
         'entities': [{'start': 2,
           'end': 4,
           'value': '空调',
           'entity': 'MajorName',
           'extractor': 'RegexExtractor'}],
         'intent_ranking': [{'name': 'faultDiag',
           'confidence': 0.8657001852989197},
          {'name': 'equipFaultDiag', 'confidence': 0.04571297764778137},
          {'name': 'inform', 'confidence': 0.028037212789058685},
          {'name': 'opinion+negtive', 'confidence': 0.010507158003747463},
          {'name': 'unknow', 'confidence': 0.006765459664165974},
          {'name': 'greet', 'confidence': 0.006672943476587534},
          {'name': 'maintenance', 'confidence': 0.004478101618587971},
          {'name': 'sysFaultDiag', 'confidence': 0.0033905007876455784},
          {'name': 'thankyou', 'confidence': 0.003203833010047674},
          {'name': 'insult', 'confidence': 0.0031317586544901133}],
         'text': '我的空调有问题'},
        'input_channel': None,
        'message_id': 'c4b6e92481ba447db80303e2ba8aa2ca',
        'metadata': {}},
       {'event': 'action',
        'timestamp': 1589357943.273126,
        'name': 'utter_faultDiag',
        'policy': 'policy_3_MappingPolicy',
        'confidence': 1},
       {'event': 'bot',
        'timestamp': 1589357943.273126,
        'text': '您要查询系统故障，还是设备故障？',
        'data': {'elements': None,
         'quick_replies': None,
         'buttons': [{'payload': '/sysFaultDiag',
           'title': '系统故障:(例如，中央空调供水温度降不下来)'},
          {'payload': '/equipFaultDiag', 'title': '设备故障:(例如，水泵喘振)'}],
         'attachment': None,
         'image': None,
         'custom': None},
        'metadata': {}},
       {'event': 'action',
        'timestamp': 1589357943.3049924,
        'name': 'action_listen',
        'policy': 'policy_3_MappingPolicy',
        'confidence': 1},
       {'event': 'user',
        'timestamp': 1589357962.350755,
        'text': '我的空调有问题',
        'parse_data': {'intent': {'name': 'faultDiag',
          'confidence': 0.8657001852989197},
         'entities': [{'start': 2,
           'end': 4,
           'value': '空调',
           'entity': 'MajorName',
           'extractor': 'RegexExtractor'}],
         'intent_ranking': [{'name': 'faultDiag',
           'confidence': 0.8657001852989197},
          {'name': 'equipFaultDiag', 'confidence': 0.04571297764778137},
          {'name': 'inform', 'confidence': 0.028037212789058685},
          {'name': 'opinion+negtive', 'confidence': 0.010507158003747463},
          {'name': 'unknow', 'confidence': 0.006765459664165974},
          {'name': 'greet', 'confidence': 0.006672943476587534},
          {'name': 'maintenance', 'confidence': 0.004478101618587971},
          {'name': 'sysFaultDiag', 'confidence': 0.0033905007876455784},
          {'name': 'thankyou', 'confidence': 0.003203833010047674},
          {'name': 'insult', 'confidence': 0.0031317586544901133}],
         'text': '我的空调有问题'},
        'input_channel': None,
        'message_id': 'c0ef6e9a2fa0483fa974e7bdcf14a65e',
        'metadata': {}},
       {'event': 'action',
        'timestamp': 1589357962.3657157,
        'name': 'utter_faultDiag',
        'policy': 'policy_3_MappingPolicy',
        'confidence': 1},
       {'event': 'bot',
        'timestamp': 1589357962.3657157,
        'text': '您要查询系统故障，还是设备故障？',
        'data': {'elements': None,
         'quick_replies': None,
         'buttons': [{'payload': '/sysFaultDiag',
           'title': '系统故障:(例如，中央空调供水温度降不下来)'},
          {'payload': '/equipFaultDiag', 'title': '设备故障:(例如，水泵喘振)'}],
         'attachment': None,
         'image': None,
         'custom': None},
        'metadata': {}},
       {'event': 'action',
        'timestamp': 1589357962.3806748,
        'name': 'action_listen',
        'policy': 'policy_3_MappingPolicy',
        'confidence': 1}],
      'latest_input_channel': None,
      'active_form': {},
      'latest_action_name': 'action_listen'}}



## 5、 Policy（策略）相关接口分析

<p>1、Rasa对话策略包括：</p>
<p>（1）规则策略：FormPolicy、FallbackPolicy、TwoStageFallbackPolicy、MemoizationPolicy、MappingPolicy、AugmentedMemoizationPolicy </p>
<p>（2）监督学习策略：KerasPolicy、TEDPolicy</p>
<p>2、继承关系</p>
<p>（1）Policy 是FallbackPolicy、MemoizationPolicy、MappingPolicy、KerasPolicy、TEDPolicy的父类；</p>
<p>（2）FallbackPolicy是TwoStageFallbackPolicy的父类；</p>
<p>（3）MemoizationPolicy是AugmentedMemoizationPolicy、FormPolicy的父类；</p>
<p>3、policy ensemble</p>
<p>3.1、PolicyEnsemble：</p>
<p>（1）聚合各个独立的Policy：FormPolicy、FallbackPolicy、TwoStageFallbackPolicy、MemoizationPolicy、MappingPolicy、AugmentedMemoizationPolicy、KerasPolicy、TEDPolicy等；</p>
<p>（2）提供策略模型与domain文件兼容性验证接口、策略模型参数打包存储和加载接口、加载状态特征器接口、策略训练接口等；</p>
<p>（3）待实现接口probabilities_using_best_policy</p>
<p>3.2、SimplePolicyEnsemble继承PolicyEnsemble</p>
<p>（1）主要功能：在PolicyEnsemble的基础上，综合各policy独立预测的next action的概率值，以及策略的优先级，得到probabilities, 和最优的policy_name</p>
<p>（2）实现了probabilities_using_best_policy：基于当前tracker预测bot的next action，根据预测概率和策略优先度来选择best policy</p>



```python
#from rasa.core.policies.ensemble import PolicyEnsemble, SimplePolicyEnsemble
#from rasa.core.policies.memoization import MemoizationPolicy
#from rasa.core.policies.policy import Policy
#@staticmethod
#def _create_ensemble(
#    policies: Union[List[Policy], PolicyEnsemble, None]
#) -> Optional[PolicyEnsemble]:
policy_ensemble = agent.policy_ensemble
print(policy_ensemble) #SimplePolicyEnsemble 
policy_ensemble.__dict__
```

    <rasa.core.policies.ensemble.SimplePolicyEnsemble object at 0x0000025FDDC58D88>
    




    {'policies': [<rasa.core.policies.two_stage_fallback.TwoStageFallbackPolicy at 0x25fddc58d48>,
      <rasa.core.policies.memoization.MemoizationPolicy at 0x26008415c88>,
      <rasa.core.policies.form_policy.FormPolicy at 0x25fddc58e48>,
      <rasa.core.policies.mapping_policy.MappingPolicy at 0x26008415a08>,
      <rasa.core.policies.keras_policy.KerasPolicy at 0x26008415ec8>],
     'date_trained': None,
     'action_fingerprints': {'action_listen': {'slots': ['EqpType',
        'Brand',
        'FtCode',
        'EqpName']},
      'deviceParameter_form': {'slots': []},
      'equipMaintenance_form': {'slots': ['EqpType',
        'requested_slot',
        'Brand',
        'EqpName']},
      'sysFaultDiag_form': {'slots': []},
      'equipFaultDiag_form': {'slots': ['requested_slot',
        'Brand',
        'FtCode',
        'EqpType',
        'EqpName']},
      'equipInstall_form': {'slots': ['EqpType',
        'requested_slot',
        'Brand',
        'EqpName']},
      'sysOperation_form': {'slots': []},
      'sysMaintenance_form': {'slots': []},
      'equipOperation_form': {'slots': ['EqpType',
        'requested_slot',
        'Brand',
        'EqpName']},
      'sysInstall_form': {'slots': []},
      'utter_slots_sysInstall': {'slots': []},
      'utter_slots_equipMaintenance': {'slots': []},
      'utter_slots_equipInstall': {'slots': []},
      'utter_slots_equipFaultDiag': {'slots': []},
      'utter_slots_equipOperation': {'slots': []},
      'utter_slots_deviceParameter': {'slots': []},
      'utter_slots_sysFaultDiag': {'slots': []},
      'utter_slots_sysMaintenance': {'slots': []},
      'utter_slots_sysOperation': {'slots': []}}}



## 6、tracker（对话跟踪器）相关接口分析

<p>主要由类DialogueStateTracker来实现对话系统状态跟踪</p>
DialogueStateTracker的主要方法：</p>
<p>1、DialogueStateTracker生成接口：从转储创建一个跟踪器 or 采用事件列表创建跟踪器
<p>2、current_state：当前对话系统状态</p>
<p>3、past_states：根据历史记录生成此跟踪器的过去状态（用于进一步产生状态向量）。</p>
<p>4、update：根据“事件”修改跟踪器的当前状态（tracker.current_state）。</p>
<p>5、export_stories_to_file将跟踪器输出为story格式，可以作为训练数据的扩充方法</p>

tracker.current_state，对话的当前状态

### 6.1、 当前对话状态tracker.current_state


```python
tracker = predict_next['tracker'] #返回dict
tracker 
#tracker.current_state的格式如下：
```




    {'sender_id': 'default',
     'slots': {'Brand': None,
      'EqpName': None,
      'EqpType': None,
      'FaultPhenomenon': None,
      'FtCode': None,
      'InstallApproach': None,
      'Maintaining': None,
      'OpApproach': None,
      'OutMark': None,
      'Param': None,
      'SysName': None,
      'feedback': None,
      'requested_slot': None},
     'latest_message': {'intent': {'name': 'faultDiag',
       'confidence': 0.8657001852989197},
      'entities': [{'start': 2,
        'end': 4,
        'value': '空调',
        'entity': 'MajorName',
        'extractor': 'RegexExtractor'}],
      'intent_ranking': [{'name': 'faultDiag', 'confidence': 0.8657001852989197},
       {'name': 'equipFaultDiag', 'confidence': 0.04571297764778137},
       {'name': 'inform', 'confidence': 0.028037212789058685},
       {'name': 'opinion+negtive', 'confidence': 0.010507158003747463},
       {'name': 'unknow', 'confidence': 0.006765459664165974},
       {'name': 'greet', 'confidence': 0.006672943476587534},
       {'name': 'maintenance', 'confidence': 0.004478101618587971},
       {'name': 'sysFaultDiag', 'confidence': 0.0033905007876455784},
       {'name': 'thankyou', 'confidence': 0.003203833010047674},
       {'name': 'insult', 'confidence': 0.0031317586544901133}],
      'text': '我的空调有问题'},
     'latest_event_time': 1589357962.3806748,
     'followup_action': None,
     'paused': False,
     'events': [{'event': 'action',
       'timestamp': 1589357942.8721497,
       'name': 'action_session_start',
       'policy': None,
       'confidence': None},
      {'event': 'session_started', 'timestamp': 1589357942.8721497},
      {'event': 'action',
       'timestamp': 1589357942.8721497,
       'name': 'action_listen',
       'policy': None,
       'confidence': None},
      {'event': 'user',
       'timestamp': 1589357943.052667,
       'text': '我的空调有问题',
       'parse_data': {'intent': {'name': 'faultDiag',
         'confidence': 0.8657001852989197},
        'entities': [{'start': 2,
          'end': 4,
          'value': '空调',
          'entity': 'MajorName',
          'extractor': 'RegexExtractor'}],
        'intent_ranking': [{'name': 'faultDiag', 'confidence': 0.8657001852989197},
         {'name': 'equipFaultDiag', 'confidence': 0.04571297764778137},
         {'name': 'inform', 'confidence': 0.028037212789058685},
         {'name': 'opinion+negtive', 'confidence': 0.010507158003747463},
         {'name': 'unknow', 'confidence': 0.006765459664165974},
         {'name': 'greet', 'confidence': 0.006672943476587534},
         {'name': 'maintenance', 'confidence': 0.004478101618587971},
         {'name': 'sysFaultDiag', 'confidence': 0.0033905007876455784},
         {'name': 'thankyou', 'confidence': 0.003203833010047674},
         {'name': 'insult', 'confidence': 0.0031317586544901133}],
        'text': '我的空调有问题'},
       'input_channel': None,
       'message_id': 'c4b6e92481ba447db80303e2ba8aa2ca',
       'metadata': {}},
      {'event': 'action',
       'timestamp': 1589357943.273126,
       'name': 'utter_faultDiag',
       'policy': 'policy_3_MappingPolicy',
       'confidence': 1},
      {'event': 'bot',
       'timestamp': 1589357943.273126,
       'text': '您要查询系统故障，还是设备故障？',
       'data': {'elements': None,
        'quick_replies': None,
        'buttons': [{'payload': '/sysFaultDiag',
          'title': '系统故障:(例如，中央空调供水温度降不下来)'},
         {'payload': '/equipFaultDiag', 'title': '设备故障:(例如，水泵喘振)'}],
        'attachment': None,
        'image': None,
        'custom': None},
       'metadata': {}},
      {'event': 'action',
       'timestamp': 1589357943.3049924,
       'name': 'action_listen',
       'policy': 'policy_3_MappingPolicy',
       'confidence': 1},
      {'event': 'user',
       'timestamp': 1589357962.350755,
       'text': '我的空调有问题',
       'parse_data': {'intent': {'name': 'faultDiag',
         'confidence': 0.8657001852989197},
        'entities': [{'start': 2,
          'end': 4,
          'value': '空调',
          'entity': 'MajorName',
          'extractor': 'RegexExtractor'}],
        'intent_ranking': [{'name': 'faultDiag', 'confidence': 0.8657001852989197},
         {'name': 'equipFaultDiag', 'confidence': 0.04571297764778137},
         {'name': 'inform', 'confidence': 0.028037212789058685},
         {'name': 'opinion+negtive', 'confidence': 0.010507158003747463},
         {'name': 'unknow', 'confidence': 0.006765459664165974},
         {'name': 'greet', 'confidence': 0.006672943476587534},
         {'name': 'maintenance', 'confidence': 0.004478101618587971},
         {'name': 'sysFaultDiag', 'confidence': 0.0033905007876455784},
         {'name': 'thankyou', 'confidence': 0.003203833010047674},
         {'name': 'insult', 'confidence': 0.0031317586544901133}],
        'text': '我的空调有问题'},
       'input_channel': None,
       'message_id': 'c0ef6e9a2fa0483fa974e7bdcf14a65e',
       'metadata': {}},
      {'event': 'action',
       'timestamp': 1589357962.3657157,
       'name': 'utter_faultDiag',
       'policy': 'policy_3_MappingPolicy',
       'confidence': 1},
      {'event': 'bot',
       'timestamp': 1589357962.3657157,
       'text': '您要查询系统故障，还是设备故障？',
       'data': {'elements': None,
        'quick_replies': None,
        'buttons': [{'payload': '/sysFaultDiag',
          'title': '系统故障:(例如，中央空调供水温度降不下来)'},
         {'payload': '/equipFaultDiag', 'title': '设备故障:(例如，水泵喘振)'}],
        'attachment': None,
        'image': None,
        'custom': None},
       'metadata': {}},
      {'event': 'action',
       'timestamp': 1589357962.3806748,
       'name': 'action_listen',
       'policy': 'policy_3_MappingPolicy',
       'confidence': 1}],
     'latest_input_channel': None,
     'active_form': {},
     'latest_action_name': 'action_listen'}



<p>tracker.current_state中包含的数据：</p>
<p>由以上可知，tracker.current_state中包含:
<p>sender_id</p>
<p>slots（插槽状态）</p>
<p>latest_message（包含意图、实体识别结果、intent_ranking、text）</p>
<p>latest_event_time、followup_action、paused </p>
<p>events（事件，从对话开始以来发生的所有事件，包含用户和bot产生的）</p>
<p>latest_input_channel、active_form、latest_action_name

### 6.2、DialogueStateTracker解析

<p>DialogueStateTracker接口。</p>
<p>agent登记当前用户信息，并返回DialogueStateTracker</p>
<p>DialogueStateTracker采用deque队列记录对话中发生的所有事件events</p>
<p>提供了保存和更新tracker.current_state各项数据的接口</p>


```python
new_tracker = await agent.log_message(uM) #返回DialogueStateTracker
print(new_tracker)
new_tracker.__dict__
```

    <rasa.core.trackers.DialogueStateTracker object at 0x0000025FE9055588>
    




    {'_max_event_history': None,
     'events': deque([<rasa.core.events.ActionExecuted at 0x25fe90b8648>,
            <rasa.core.events.SessionStarted at 0x25fe90b8288>,
            <rasa.core.events.ActionExecuted at 0x25fe90b8d88>,
            <rasa.core.events.UserUttered at 0x25fe90b8dc8>,
            <rasa.core.events.ActionExecuted at 0x25fe90b8f88>,
            BotUttered('您要查询系统故障，还是设备故障？', {"elements": null, "quick_replies": null, "buttons": [{"payload": "/sysFaultDiag", "title": "\u7cfb\u7edf\u6545\u969c:(\u4f8b\u5982\uff0c\u4e2d\u592e\u7a7a\u8c03\u4f9b\u6c34\u6e29\u5ea6\u964d\u4e0d\u4e0b\u6765)"}, {"payload": "/equipFaultDiag", "title": "\u8bbe\u5907\u6545\u969c:(\u4f8b\u5982\uff0c\u6c34\u6cf5\u5598\u632f)"}], "attachment": null, "image": null, "custom": null}, {}, 1589357943.273126),
            <rasa.core.events.ActionExecuted at 0x25fe90b8808>,
            <rasa.core.events.UserUttered at 0x25fe90b8708>,
            <rasa.core.events.ActionExecuted at 0x25fe90b8d08>,
            BotUttered('您要查询系统故障，还是设备故障？', {"elements": null, "quick_replies": null, "buttons": [{"payload": "/sysFaultDiag", "title": "\u7cfb\u7edf\u6545\u969c:(\u4f8b\u5982\uff0c\u4e2d\u592e\u7a7a\u8c03\u4f9b\u6c34\u6e29\u5ea6\u964d\u4e0d\u4e0b\u6765)"}, {"payload": "/equipFaultDiag", "title": "\u8bbe\u5907\u6545\u969c:(\u4f8b\u5982\uff0c\u6c34\u6cf5\u5598\u632f)"}], "attachment": null, "image": null, "custom": null}, {}, 1589357962.3657157),
            <rasa.core.events.ActionExecuted at 0x25fe90b8c88>,
            <rasa.core.events.UserUttered at 0x25fe9060b48>]),
     'sender_id': 'default',
     'slots': {'Brand': <UnfeaturizedSlot(Brand: None)>,
      'EqpName': <UnfeaturizedSlot(EqpName: None)>,
      'EqpType': <UnfeaturizedSlot(EqpType: None)>,
      'FaultPhenomenon': <UnfeaturizedSlot(FaultPhenomenon: None)>,
      'FtCode': <UnfeaturizedSlot(FtCode: None)>,
      'InstallApproach': <UnfeaturizedSlot(InstallApproach: None)>,
      'Maintaining': <UnfeaturizedSlot(Maintaining: None)>,
      'OpApproach': <UnfeaturizedSlot(OpApproach: None)>,
      'OutMark': <BooleanSlot(OutMark: None)>,
      'Param': <UnfeaturizedSlot(Param: None)>,
      'SysName': <UnfeaturizedSlot(SysName: None)>,
      'feedback': <UnfeaturizedSlot(feedback: None)>,
      'requested_slot': <UnfeaturizedSlot(requested_slot: None)>},
     '_paused': False,
     'followup_action': None,
     'latest_action_name': 'action_listen',
     'latest_message': <rasa.core.events.UserUttered at 0x25fe9060b48>,
     'latest_bot_utterance': BotUttered('您要查询系统故障，还是设备故障？', {"elements": null, "quick_replies": null, "buttons": [{"payload": "/sysFaultDiag", "title": "\u7cfb\u7edf\u6545\u969c:(\u4f8b\u5982\uff0c\u4e2d\u592e\u7a7a\u8c03\u4f9b\u6c34\u6e29\u5ea6\u964d\u4e0d\u4e0b\u6765)"}, {"payload": "/equipFaultDiag", "title": "\u8bbe\u5907\u6545\u969c:(\u4f8b\u5982\uff0c\u6c34\u6cf5\u5598\u632f)"}], "attachment": null, "image": null, "custom": null}, {}, 1589357962.3657157),
     'active_form': {}}



### 6.3、对话状态向量生成过程

由tarcker产生对话状态向量，作为监督学习策略（如TEDPolicy、KerasPolicy）的输入


```python
#它用于TrackerFeaturizer的_create_states方法中，是产生状态向量的中间过程，
#得到一个deque包含n个frozenset，每个代表一轮对话（包含上一个系统动作，识别出的实体，top_10意图）
#frozenset中包括top_10 intent(用置信度表征),
states = new_tracker.past_states(domain)
states
```




    deque([frozenset(),
           frozenset({('entity_MajorName', 1.0),
                      ('intent_equipFaultDiag', 0.04571297764778137),
                      ('intent_faultDiag', 0.8657001852989197),
                      ('intent_greet', 0.006672943476587534),
                      ('intent_inform', 0.028037212789058685),
                      ('intent_insult', 0.0031317586544901133),
                      ('intent_maintenance', 0.004478101618587971),
                      ('intent_opinion+negtive', 0.010507158003747463),
                      ('intent_sysFaultDiag', 0.0033905007876455784),
                      ('intent_thankyou', 0.003203833010047674),
                      ('intent_unknow', 0.006765459664165974),
                      ('prev_action_listen', 1.0)}),
           frozenset({('entity_MajorName', 1.0),
                      ('intent_equipFaultDiag', 0.04571297764778137),
                      ('intent_faultDiag', 0.8657001852989197),
                      ('intent_greet', 0.006672943476587534),
                      ('intent_inform', 0.028037212789058685),
                      ('intent_insult', 0.0031317586544901133),
                      ('intent_maintenance', 0.004478101618587971),
                      ('intent_opinion+negtive', 0.010507158003747463),
                      ('intent_sysFaultDiag', 0.0033905007876455784),
                      ('intent_thankyou', 0.003203833010047674),
                      ('intent_unknow', 0.006765459664165974),
                      ('prev_utter_faultDiag', 1.0)}),
           frozenset({('entity_MajorName', 1.0),
                      ('intent_equipFaultDiag', 0.04571297764778137),
                      ('intent_faultDiag', 0.8657001852989197),
                      ('intent_greet', 0.006672943476587534),
                      ('intent_inform', 0.028037212789058685),
                      ('intent_insult', 0.0031317586544901133),
                      ('intent_maintenance', 0.004478101618587971),
                      ('intent_opinion+negtive', 0.010507158003747463),
                      ('intent_sysFaultDiag', 0.0033905007876455784),
                      ('intent_thankyou', 0.003203833010047674),
                      ('intent_unknow', 0.006765459664165974),
                      ('prev_action_listen', 1.0)}),
           frozenset({('entity_MajorName', 1.0),
                      ('intent_equipFaultDiag', 0.04571297764778137),
                      ('intent_faultDiag', 0.8657001852989197),
                      ('intent_greet', 0.006672943476587534),
                      ('intent_inform', 0.028037212789058685),
                      ('intent_insult', 0.0031317586544901133),
                      ('intent_maintenance', 0.004478101618587971),
                      ('intent_opinion+negtive', 0.010507158003747463),
                      ('intent_sysFaultDiag', 0.0033905007876455784),
                      ('intent_thankyou', 0.003203833010047674),
                      ('intent_unknow', 0.006765459664165974),
                      ('prev_utter_faultDiag', 1.0)}),
           frozenset({('entity_MajorName', 1.0),
                      ('intent_equipFaultDiag', 0.04571297764778137),
                      ('intent_faultDiag', 0.8657001852989197),
                      ('intent_greet', 0.006672943476587534),
                      ('intent_inform', 0.028037212789058685),
                      ('intent_insult', 0.0031317586544901133),
                      ('intent_maintenance', 0.004478101618587971),
                      ('intent_opinion+negtive', 0.010507158003747463),
                      ('intent_sysFaultDiag', 0.0033905007876455784),
                      ('intent_thankyou', 0.003203833010047674),
                      ('intent_unknow', 0.006765459664165974),
                      ('prev_action_listen', 1.0)})])



new_tracker.past_states方法的底层实现方法，来源于Domain类


```python
generated_states = domain.states_for_tracker_history(new_tracker)
generated_states
#domain.states_for_tracker_history等价于：
generated_states_1 = [domain.get_active_states(tr) for tr in new_tracker.generate_all_prior_trackers()]
generated_states,generated_states_1
```




    ([{},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_action_listen': 1.0},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_utter_faultDiag': 1.0},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_action_listen': 1.0},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_utter_faultDiag': 1.0},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_action_listen': 1.0}],
     [{},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_action_listen': 1.0},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_utter_faultDiag': 1.0},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_action_listen': 1.0},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_utter_faultDiag': 1.0},
      {'entity_MajorName': 1.0,
       'intent_faultDiag': 0.8657001852989197,
       'intent_equipFaultDiag': 0.04571297764778137,
       'intent_inform': 0.028037212789058685,
       'intent_opinion+negtive': 0.010507158003747463,
       'intent_unknow': 0.006765459664165974,
       'intent_greet': 0.006672943476587534,
       'intent_maintenance': 0.004478101618587971,
       'intent_sysFaultDiag': 0.0033905007876455784,
       'intent_thankyou': 0.003203833010047674,
       'intent_insult': 0.0031317586544901133,
       'prev_action_listen': 1.0}])




```python
#tracker_store 
#存储DialogueStateTracker的方式
print(agent.tracker_store)
print(agent.tracker_store.__dict__)
inMemoryTrckerStore = agent.tracker_store.__dict__['_tracker_store']
inMemoryTrckerStore.retrieve('defualt')
inMemoryTrckerStore.__dict__,#inMemoryTrckerStore.keys() #dict_keys(['default']),
```

    <rasa.core.tracker_store.FailSafeTrackerStore object at 0x0000026009D9CD88>
    {'_fallback_tracker_store': None, '_tracker_store': <rasa.core.tracker_store.InMemoryTrackerStore object at 0x0000026009D9C808>, '_on_tracker_store_error': None, 'event_broker': None, 'max_event_history': None}
    




    ({'store': {'default': '{"events": [{"event": "action", "timestamp": 1589357942.8721497, "name": "action_session_start", "policy": null, "confidence": null}, {"event": "session_started", "timestamp": 1589357942.8721497}, {"event": "action", "timestamp": 1589357942.8721497, "name": "action_listen", "policy": null, "confidence": null}, {"event": "user", "timestamp": 1589357943.052667, "text": "\\u6211\\u7684\\u7a7a\\u8c03\\u6709\\u95ee\\u9898", "parse_data": {"intent": {"name": "faultDiag", "confidence": 0.8657001852989197}, "entities": [{"start": 2, "end": 4, "value": "\\u7a7a\\u8c03", "entity": "MajorName", "extractor": "RegexExtractor"}], "intent_ranking": [{"name": "faultDiag", "confidence": 0.8657001852989197}, {"name": "equipFaultDiag", "confidence": 0.04571297764778137}, {"name": "inform", "confidence": 0.028037212789058685}, {"name": "opinion+negtive", "confidence": 0.010507158003747463}, {"name": "unknow", "confidence": 0.006765459664165974}, {"name": "greet", "confidence": 0.006672943476587534}, {"name": "maintenance", "confidence": 0.004478101618587971}, {"name": "sysFaultDiag", "confidence": 0.0033905007876455784}, {"name": "thankyou", "confidence": 0.003203833010047674}, {"name": "insult", "confidence": 0.0031317586544901133}], "text": "\\u6211\\u7684\\u7a7a\\u8c03\\u6709\\u95ee\\u9898"}, "input_channel": null, "message_id": "c4b6e92481ba447db80303e2ba8aa2ca", "metadata": {}}, {"event": "action", "timestamp": 1589357943.273126, "name": "utter_faultDiag", "policy": "policy_3_MappingPolicy", "confidence": 1}, {"event": "bot", "timestamp": 1589357943.273126, "text": "\\u60a8\\u8981\\u67e5\\u8be2\\u7cfb\\u7edf\\u6545\\u969c\\uff0c\\u8fd8\\u662f\\u8bbe\\u5907\\u6545\\u969c\\uff1f", "data": {"elements": null, "quick_replies": null, "buttons": [{"payload": "/sysFaultDiag", "title": "\\u7cfb\\u7edf\\u6545\\u969c:(\\u4f8b\\u5982\\uff0c\\u4e2d\\u592e\\u7a7a\\u8c03\\u4f9b\\u6c34\\u6e29\\u5ea6\\u964d\\u4e0d\\u4e0b\\u6765)"}, {"payload": "/equipFaultDiag", "title": "\\u8bbe\\u5907\\u6545\\u969c:(\\u4f8b\\u5982\\uff0c\\u6c34\\u6cf5\\u5598\\u632f)"}], "attachment": null, "image": null, "custom": null}, "metadata": {}}, {"event": "action", "timestamp": 1589357943.3049924, "name": "action_listen", "policy": "policy_3_MappingPolicy", "confidence": 1}, {"event": "user", "timestamp": 1589357962.350755, "text": "\\u6211\\u7684\\u7a7a\\u8c03\\u6709\\u95ee\\u9898", "parse_data": {"intent": {"name": "faultDiag", "confidence": 0.8657001852989197}, "entities": [{"start": 2, "end": 4, "value": "\\u7a7a\\u8c03", "entity": "MajorName", "extractor": "RegexExtractor"}], "intent_ranking": [{"name": "faultDiag", "confidence": 0.8657001852989197}, {"name": "equipFaultDiag", "confidence": 0.04571297764778137}, {"name": "inform", "confidence": 0.028037212789058685}, {"name": "opinion+negtive", "confidence": 0.010507158003747463}, {"name": "unknow", "confidence": 0.006765459664165974}, {"name": "greet", "confidence": 0.006672943476587534}, {"name": "maintenance", "confidence": 0.004478101618587971}, {"name": "sysFaultDiag", "confidence": 0.0033905007876455784}, {"name": "thankyou", "confidence": 0.003203833010047674}, {"name": "insult", "confidence": 0.0031317586544901133}], "text": "\\u6211\\u7684\\u7a7a\\u8c03\\u6709\\u95ee\\u9898"}, "input_channel": null, "message_id": "c0ef6e9a2fa0483fa974e7bdcf14a65e", "metadata": {}}, {"event": "action", "timestamp": 1589357962.3657157, "name": "utter_faultDiag", "policy": "policy_3_MappingPolicy", "confidence": 1}, {"event": "bot", "timestamp": 1589357962.3657157, "text": "\\u60a8\\u8981\\u67e5\\u8be2\\u7cfb\\u7edf\\u6545\\u969c\\uff0c\\u8fd8\\u662f\\u8bbe\\u5907\\u6545\\u969c\\uff1f", "data": {"elements": null, "quick_replies": null, "buttons": [{"payload": "/sysFaultDiag", "title": "\\u7cfb\\u7edf\\u6545\\u969c:(\\u4f8b\\u5982\\uff0c\\u4e2d\\u592e\\u7a7a\\u8c03\\u4f9b\\u6c34\\u6e29\\u5ea6\\u964d\\u4e0d\\u4e0b\\u6765)"}, {"payload": "/equipFaultDiag", "title": "\\u8bbe\\u5907\\u6545\\u969c:(\\u4f8b\\u5982\\uff0c\\u6c34\\u6cf5\\u5598\\u632f)"}], "attachment": null, "image": null, "custom": null}, "metadata": {}}, {"event": "action", "timestamp": 1589357962.3806748, "name": "action_listen", "policy": "policy_3_MappingPolicy", "confidence": 1}, {"event": "user", "timestamp": 1589358162.9750464, "text": "\\u6211\\u7684\\u7a7a\\u8c03\\u6709\\u95ee\\u9898", "parse_data": {"intent": {"name": "faultDiag", "confidence": 0.8657001852989197}, "entities": [{"start": 2, "end": 4, "value": "\\u7a7a\\u8c03", "entity": "MajorName", "extractor": "RegexExtractor"}], "intent_ranking": [{"name": "faultDiag", "confidence": 0.8657001852989197}, {"name": "equipFaultDiag", "confidence": 0.04571297764778137}, {"name": "inform", "confidence": 0.028037212789058685}, {"name": "opinion+negtive", "confidence": 0.010507158003747463}, {"name": "unknow", "confidence": 0.006765459664165974}, {"name": "greet", "confidence": 0.006672943476587534}, {"name": "maintenance", "confidence": 0.004478101618587971}, {"name": "sysFaultDiag", "confidence": 0.0033905007876455784}, {"name": "thankyou", "confidence": 0.003203833010047674}, {"name": "insult", "confidence": 0.0031317586544901133}], "text": "\\u6211\\u7684\\u7a7a\\u8c03\\u6709\\u95ee\\u9898"}, "input_channel": null, "message_id": "c0ef6e9a2fa0483fa974e7bdcf14a65e", "metadata": {}}], "name": "default"}'},
      'domain': <rasa.core.domain.Domain at 0x26009d8dd08>,
      'event_broker': None,
      'max_event_history': None},)



将DialogueStateTracker带入MaxHistoryTrackerFeaturizer等特征器，生成状态向量


```python
binarySingleStateFeaturizer = BinarySingleStateFeaturizer()
maxHistoryTrackerFeaturizer = MaxHistoryTrackerFeaturizer(
        state_featurizer=binarySingleStateFeaturizer,max_history='5')
print(maxHistoryTrackerFeaturizer._create_states(new_tracker, domain))
maxHistoryTrackerFeaturizer._create_states(new_tracker, domain,is_binary_training=True)
```

    [{}, {'prev_action_listen': 1.0, 'entity_MajorName': 1.0, 'intent_faultDiag': 1.0}, {'entity_MajorName': 1.0, 'prev_utter_faultDiag': 1.0, 'intent_faultDiag': 1.0}, {'prev_action_listen': 1.0, 'entity_MajorName': 1.0, 'intent_faultDiag': 1.0}, {'entity_MajorName': 1.0, 'prev_utter_faultDiag': 1.0, 'intent_faultDiag': 1.0}, {'prev_action_listen': 1.0, 'entity_MajorName': 1.0, 'intent_faultDiag': 1.0}]
    




    [{},
     {'intent_equipFaultDiag': 0.04571297764778137,
      'intent_opinion+negtive': 0.010507158003747463,
      'intent_inform': 0.028037212789058685,
      'intent_insult': 0.0031317586544901133,
      'intent_sysFaultDiag': 0.0033905007876455784,
      'intent_thankyou': 0.003203833010047674,
      'prev_action_listen': 1.0,
      'entity_MajorName': 1.0,
      'intent_unknow': 0.006765459664165974,
      'intent_faultDiag': 0.8657001852989197,
      'intent_greet': 0.006672943476587534,
      'intent_maintenance': 0.004478101618587971},
     {'intent_equipFaultDiag': 0.04571297764778137,
      'intent_opinion+negtive': 0.010507158003747463,
      'intent_inform': 0.028037212789058685,
      'intent_insult': 0.0031317586544901133,
      'intent_sysFaultDiag': 0.0033905007876455784,
      'intent_thankyou': 0.003203833010047674,
      'entity_MajorName': 1.0,
      'intent_unknow': 0.006765459664165974,
      'prev_utter_faultDiag': 1.0,
      'intent_faultDiag': 0.8657001852989197,
      'intent_greet': 0.006672943476587534,
      'intent_maintenance': 0.004478101618587971},
     {'intent_equipFaultDiag': 0.04571297764778137,
      'intent_opinion+negtive': 0.010507158003747463,
      'intent_inform': 0.028037212789058685,
      'intent_insult': 0.0031317586544901133,
      'intent_sysFaultDiag': 0.0033905007876455784,
      'intent_thankyou': 0.003203833010047674,
      'prev_action_listen': 1.0,
      'entity_MajorName': 1.0,
      'intent_unknow': 0.006765459664165974,
      'intent_faultDiag': 0.8657001852989197,
      'intent_greet': 0.006672943476587534,
      'intent_maintenance': 0.004478101618587971},
     {'intent_equipFaultDiag': 0.04571297764778137,
      'intent_opinion+negtive': 0.010507158003747463,
      'intent_inform': 0.028037212789058685,
      'intent_insult': 0.0031317586544901133,
      'intent_sysFaultDiag': 0.0033905007876455784,
      'intent_thankyou': 0.003203833010047674,
      'entity_MajorName': 1.0,
      'intent_unknow': 0.006765459664165974,
      'prev_utter_faultDiag': 1.0,
      'intent_faultDiag': 0.8657001852989197,
      'intent_greet': 0.006672943476587534,
      'intent_maintenance': 0.004478101618587971},
     {'intent_equipFaultDiag': 0.04571297764778137,
      'intent_opinion+negtive': 0.010507158003747463,
      'intent_inform': 0.028037212789058685,
      'intent_insult': 0.0031317586544901133,
      'intent_sysFaultDiag': 0.0033905007876455784,
      'intent_thankyou': 0.003203833010047674,
      'prev_action_listen': 1.0,
      'entity_MajorName': 1.0,
      'intent_unknow': 0.006765459664165974,
      'intent_faultDiag': 0.8657001852989197,
      'intent_greet': 0.006672943476587534,
      'intent_maintenance': 0.004478101618587971}]



## 7、LockStore分析


```python
print(uM, uM.sender_id)
lock_store = agent.lock_store
print(lock_store, lock_store.__dict__)

print(lock_store.get_lock(uM.sender_id))
```

    <rasa.core.channels.channel.UserMessage object at 0x0000025FE70072C8> default
    <rasa.core.lock_store.InMemoryLockStore object at 0x0000026009D9C488> {'conversation_locks': {}}
    None
    


```python
ticketLock = lock_store.create_lock(uM.sender_id)
print(ticketLock, ticketLock.__dict__)
lock_store.save_lock(ticketLock) #必须保存才能get_lock
print(lock_store.get_lock(uM.sender_id))
ticketLock.issue_ticket(10.0)
lock_store.is_someone_waiting(uM.sender_id)
```

    <rasa.core.lock.TicketLock object at 0x0000025FE7004F48> {'conversation_id': 'default', 'tickets': deque([])}
    <rasa.core.lock.TicketLock object at 0x0000025FE7004F48>
    




    True



## 8、nlu解析（数据、训练）

### 8.1、nlu模型组成各个部件（components）拆解


```python
ms_text = '我的空调系统出现故障'
uM = UserMessage(ms_text)
uM.__dict__

message = Message(ms_text)
print(message.__dict__)

components.ComponentBuilder()
print(NLU_config.component_names)

trainer = Trainer(NLU_config)
print(trainer.__dict__)
print(trainer.pipeline)

#tokenizer
jieba_tokenizer = trainer.pipeline[0]
print(jieba_tokenizer.__dict__)
tokens = jieba_tokenizer.tokenize(message, 'text')
print([token.__dict__ for token in tokens])
#jieba_tokenizer.process(message, {}) #有问题？？？
#jieba_tokenizer.train()

#extractor
kashgariExtractor = trainer.pipeline[1]
print(kashgariExtractor.__dict__)
training_data = await file_importer.get_nlu_data()
print(training_data.__dict__)
training_data.nlg_stories
training_data.lookup_tables
training_data.nlg_stories
training_data.training_examples[1000].__dict__
kashgariExtractor.train(training_data, '') #可以训练
kashgariExtractor.extract_entities(message)
```

    {'text': '我的空调系统出现故障', 'time': None, 'data': {}, 'output_properties': set()}
    ['JiebaTokenizer', 'KashgariEntityExtractor', 'RegexExtractor', 'KashgariIntentClassifier']
    

    WARNING:root:Sequence length will auto set at 95% of sequence length
    WARNING:root:Model will be built until sequence length is determined
    WARNING:root:Sequence length will auto set at 95% of sequence length
    WARNING:root:Model will be built until sequence length is determined
    

    {'config': <rasa.nlu.config.RasaNLUModelConfig object at 0x000002071B525B48>, 'skip_validation': False, 'training_data': None, 'pipeline': [<rasa.nlu.tokenizers.jieba_tokenizer.JiebaTokenizer object at 0x000002073CD69608>, <rasa.nlu.extractors.kashgari_entity_extractor.KashgariEntityExtractor object at 0x000002073CDCA408>, <rasa.nlu.extractors.regex_extractor.RegexExtractor object at 0x000002079870FB48>, <rasa.nlu.classifiers.kashgari_intent_classifier.KashgariIntentClassifier object at 0x000002079870FA48>]}
    [<rasa.nlu.tokenizers.jieba_tokenizer.JiebaTokenizer object at 0x000002073CD69608>, <rasa.nlu.extractors.kashgari_entity_extractor.KashgariEntityExtractor object at 0x000002073CDCA408>, <rasa.nlu.extractors.regex_extractor.RegexExtractor object at 0x000002079870FB48>, <rasa.nlu.classifiers.kashgari_intent_classifier.KashgariIntentClassifier object at 0x000002079870FA48>]
    {'component_config': {'dictionary_path': None, 'intent_tokenization_flag': False, 'intent_split_symbol': '_', 'name': 'JiebaTokenizer'}, 'partial_processing_pipeline': None, 'partial_processing_context': None, 'intent_tokenization_flag': False, 'intent_split_symbol': '_', 'dictionary_path': None}
    [{'text': '我', 'start': 0, 'end': 1, 'data': {}, 'lemma': '我'}, {'text': '的', 'start': 1, 'end': 2, 'data': {}, 'lemma': '的'}, {'text': '空调', 'start': 2, 'end': 4, 'data': {}, 'lemma': '空调'}, {'text': '系统', 'start': 4, 'end': 6, 'data': {}, 'lemma': '系统'}, {'text': '出现', 'start': 6, 'end': 8, 'data': {}, 'lemma': '出现'}, {'text': '故障', 'start': 8, 'end': 10, 'data': {}, 'lemma': '故障'}]
    {'component_config': {'bert_model_path': 'chinese_L-12_H-768_A-12', 'sequence_length': 'auto', 'layer_nums': 4, 'trainable': False, 'labeling_model': 'BiLSTM_Model', 'epochs': 10, 'batch_size': 32, 'validation_split': 0.2, 'patience': 5, 'factor': 0.5, 'verbose': 1, 'use_cudnn_cell': False, 'name': 'KashgariEntityExtractor'}, 'partial_processing_pipeline': None, 'partial_processing_context': None, 'labeling_model': 'BiLSTM_Model', 'bert_embedding': <<class 'kashgari.embeddings.bert_embedding.BERTEmbedding'> seq_len: auto>, 'model': None}
    {'training_examples': [<rasa.nlu.training_data.message.Message object at 0x0000020798BAF948>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAF808>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAF908>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFA88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFB88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFAC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFBC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFB08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFCC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFD48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFD88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFDC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFE08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFE88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFEC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFF88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BAFFC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0048>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0088>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB00C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0108>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0148>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0188>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB01C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0208>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0248>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0308>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB03C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0488>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB05C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB06C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB07C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0888>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0988>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB0DC8>, <rasa.nlu.training_data.message.Message object at 0x000002073CDDDA08>, <rasa.nlu.training_data.message.Message object at 0x000002073CDDDA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB7248>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB7488>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB76C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB7908>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB7D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB7B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB7D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BB7088>, <rasa.nlu.training_data.message.Message object at 0x0000020798A84388>, <rasa.nlu.training_data.message.Message object at 0x0000020798A84708>, <rasa.nlu.training_data.message.Message object at 0x0000020798A843C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A84C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798A84908>, <rasa.nlu.training_data.message.Message object at 0x0000020798A84C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798A910C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A91588>, <rasa.nlu.training_data.message.Message object at 0x0000020798A912C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A91988>, <rasa.nlu.training_data.message.Message object at 0x0000020798A91748>, <rasa.nlu.training_data.message.Message object at 0x0000020798A91D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798A91B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798A91F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798A98348>, <rasa.nlu.training_data.message.Message object at 0x0000020798A985C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A98388>, <rasa.nlu.training_data.message.Message object at 0x0000020798A989C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A98708>, <rasa.nlu.training_data.message.Message object at 0x0000020798A98DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A98B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798A98E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BBF148>, <rasa.nlu.training_data.message.Message object at 0x0000020798BBF088>, <rasa.nlu.training_data.message.Message object at 0x0000020798BBF788>, <rasa.nlu.training_data.message.Message object at 0x0000020798BBFA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BBF7C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BBFF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BBFE08>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA3088>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA30C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA3588>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA33C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA3888>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA36C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA3B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA39C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA3E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA3CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AA3FC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A9D348>, <rasa.nlu.training_data.message.Message object at 0x0000020798A9D548>, <rasa.nlu.training_data.message.Message object at 0x0000020798A9D388>, <rasa.nlu.training_data.message.Message object at 0x0000020798A9D848>, <rasa.nlu.training_data.message.Message object at 0x0000020798A9DA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798A9DC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798A9DE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798A9DD88>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAF148>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAF548>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAF748>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAF948>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAFB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAFD48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAFF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAFF88>, <rasa.nlu.training_data.message.Message object at 0x0000020798ABC048>, <rasa.nlu.training_data.message.Message object at 0x0000020798ABC248>, <rasa.nlu.training_data.message.Message object at 0x0000020798ABC448>, <rasa.nlu.training_data.message.Message object at 0x0000020798ABC648>, <rasa.nlu.training_data.message.Message object at 0x0000020798ABC848>, <rasa.nlu.training_data.message.Message object at 0x0000020798ABCA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798ABCC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC90C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC9408>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC9808>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC9B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC9E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC9D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798ACE3C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798ACE748>, <rasa.nlu.training_data.message.Message object at 0x0000020798ACEAC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798ACED48>, <rasa.nlu.training_data.message.Message object at 0x0000020798ACEC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AD3308>, <rasa.nlu.training_data.message.Message object at 0x0000020798AD3748>, <rasa.nlu.training_data.message.Message object at 0x0000020798AD3A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AD3CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AD3FC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADF088>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADF0C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADF348>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADF548>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADF748>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADF948>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADFBC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AE60C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AE6108>, <rasa.nlu.training_data.message.Message object at 0x0000020798AE6388>, <rasa.nlu.training_data.message.Message object at 0x0000020798AE6608>, <rasa.nlu.training_data.message.Message object at 0x0000020798AE6908>, <rasa.nlu.training_data.message.Message object at 0x0000020798AE6B08>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC3048>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC3088>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC3288>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC3488>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC3CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AC3BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF00C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF0108>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF0308>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF0508>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF0708>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF0908>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF0B08>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF0D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF7248>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF7548>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF7748>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF7948>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF7B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF7D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF7F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AF7F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B05348>, <rasa.nlu.training_data.message.Message object at 0x0000020798B05248>, <rasa.nlu.training_data.message.Message object at 0x0000020798B054C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B057C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B05D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B0B348>, <rasa.nlu.training_data.message.Message object at 0x0000020798B0B288>, <rasa.nlu.training_data.message.Message object at 0x0000020798B0B5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B0B948>, <rasa.nlu.training_data.message.Message object at 0x0000020798B0BC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B10308>, <rasa.nlu.training_data.message.Message object at 0x0000020798B10188>, <rasa.nlu.training_data.message.Message object at 0x0000020798B10488>, <rasa.nlu.training_data.message.Message object at 0x0000020798B10748>, <rasa.nlu.training_data.message.Message object at 0x0000020798B109C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B10C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B170C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B17108>, <rasa.nlu.training_data.message.Message object at 0x0000020798B17408>, <rasa.nlu.training_data.message.Message object at 0x0000020798B17708>, <rasa.nlu.training_data.message.Message object at 0x0000020798B17988>, <rasa.nlu.training_data.message.Message object at 0x0000020798B17C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B17F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B1B048>, <rasa.nlu.training_data.message.Message object at 0x0000020798B1B2C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B1B548>, <rasa.nlu.training_data.message.Message object at 0x0000020798B1B7C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B1BA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B1BDC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AFF288>, <rasa.nlu.training_data.message.Message object at 0x0000020798AFF1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AFF3C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AFF5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AFFEC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AFFF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B2F408>, <rasa.nlu.training_data.message.Message object at 0x0000020798B2F248>, <rasa.nlu.training_data.message.Message object at 0x0000020798B2F5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B2F888>, <rasa.nlu.training_data.message.Message object at 0x0000020798B2FB08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B2FE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B35048>, <rasa.nlu.training_data.message.Message object at 0x0000020798B35348>, <rasa.nlu.training_data.message.Message object at 0x0000020798B35648>, <rasa.nlu.training_data.message.Message object at 0x0000020798B35848>, <rasa.nlu.training_data.message.Message object at 0x0000020798B35D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B3A2C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B3A188>, <rasa.nlu.training_data.message.Message object at 0x0000020798B3A488>, <rasa.nlu.training_data.message.Message object at 0x0000020798B3A708>, <rasa.nlu.training_data.message.Message object at 0x0000020798B3A908>, <rasa.nlu.training_data.message.Message object at 0x0000020798B3AC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B3AF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B29648>, <rasa.nlu.training_data.message.Message object at 0x0000020798B29448>, <rasa.nlu.training_data.message.Message object at 0x0000020798B297C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B29A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B29288>, <rasa.nlu.training_data.message.Message object at 0x0000020798B4B048>, <rasa.nlu.training_data.message.Message object at 0x0000020798B4B248>, <rasa.nlu.training_data.message.Message object at 0x0000020798B4B448>, <rasa.nlu.training_data.message.Message object at 0x0000020798B4B648>, <rasa.nlu.training_data.message.Message object at 0x0000020798B4B848>, <rasa.nlu.training_data.message.Message object at 0x0000020798B4BA08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B4BC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B4BF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B47108>, <rasa.nlu.training_data.message.Message object at 0x0000020798B47308>, <rasa.nlu.training_data.message.Message object at 0x0000020798B47588>, <rasa.nlu.training_data.message.Message object at 0x0000020798B47788>, <rasa.nlu.training_data.message.Message object at 0x0000020798B47EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B47F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B59448>, <rasa.nlu.training_data.message.Message object at 0x0000020798B59648>, <rasa.nlu.training_data.message.Message object at 0x0000020798B59848>, <rasa.nlu.training_data.message.Message object at 0x0000020798B59AC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B59D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B59C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B5E288>, <rasa.nlu.training_data.message.Message object at 0x0000020798B5E6C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B5EAC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B5ED48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B5EC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B532C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B53788>, <rasa.nlu.training_data.message.Message object at 0x0000020798B53988>, <rasa.nlu.training_data.message.Message object at 0x0000020798B53B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B53E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B53D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC6188>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC6488>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC6688>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC6888>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC6A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC6C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC6E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC6DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BCD1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BCD4C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BCD6C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BCD8C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BCDAC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BCDCC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BCDEC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BCDF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BD3048>, <rasa.nlu.training_data.message.Message object at 0x0000020798BD3248>, <rasa.nlu.training_data.message.Message object at 0x0000020798BD3448>, <rasa.nlu.training_data.message.Message object at 0x0000020798BD3648>, <rasa.nlu.training_data.message.Message object at 0x0000020798BD3848>, <rasa.nlu.training_data.message.Message object at 0x0000020798BD3A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BD3C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC0188>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC0608>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC0548>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC0748>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC0948>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC0B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BC0E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1148>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1388>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1508>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1688>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1808>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1988>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1B08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE1FC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE62C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE6448>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE65C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE6748>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE68C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE6A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE6BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE6D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE6EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BE6E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFC148>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFC388>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFC508>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFC688>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFC808>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFC988>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFCC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFCE88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BFCF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C003C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C00648>, <rasa.nlu.training_data.message.Message object at 0x0000020798C008C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C00B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C00DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C00D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C08188>, <rasa.nlu.training_data.message.Message object at 0x0000020798C08548>, <rasa.nlu.training_data.message.Message object at 0x0000020798C087C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C08A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C08CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C08F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C0C0C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C0C488>, <rasa.nlu.training_data.message.Message object at 0x0000020798C0C708>, <rasa.nlu.training_data.message.Message object at 0x0000020798C0C988>, <rasa.nlu.training_data.message.Message object at 0x0000020798C0CC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C0CE88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C0CF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C103C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C10648>, <rasa.nlu.training_data.message.Message object at 0x0000020798C108C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C10B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C10DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C10D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C041C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C04588>, <rasa.nlu.training_data.message.Message object at 0x0000020798C04788>, <rasa.nlu.training_data.message.Message object at 0x0000020798C049C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C04C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C04EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C04F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C14308>, <rasa.nlu.training_data.message.Message object at 0x0000020798C14288>, <rasa.nlu.training_data.message.Message object at 0x0000020798C14508>, <rasa.nlu.training_data.message.Message object at 0x0000020798C14788>, <rasa.nlu.training_data.message.Message object at 0x0000020798C14A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C14C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C1E1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C1E148>, <rasa.nlu.training_data.message.Message object at 0x0000020798C1E2C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C1E448>, <rasa.nlu.training_data.message.Message object at 0x0000020798C1E5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C1E748>, <rasa.nlu.training_data.message.Message object at 0x0000020798C1EA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C24048>, <rasa.nlu.training_data.message.Message object at 0x0000020798C24088>, <rasa.nlu.training_data.message.Message object at 0x0000020798C24388>, <rasa.nlu.training_data.message.Message object at 0x0000020798C24688>, <rasa.nlu.training_data.message.Message object at 0x0000020798C24988>, <rasa.nlu.training_data.message.Message object at 0x0000020798C24E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C29388>, <rasa.nlu.training_data.message.Message object at 0x0000020798C292C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C295C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C298C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C29B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C2F048>, <rasa.nlu.training_data.message.Message object at 0x0000020798C2F088>, <rasa.nlu.training_data.message.Message object at 0x0000020798C2F308>, <rasa.nlu.training_data.message.Message object at 0x0000020798C2F588>, <rasa.nlu.training_data.message.Message object at 0x0000020798C2F808>, <rasa.nlu.training_data.message.Message object at 0x0000020798C2FA88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C2FE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C39348>, <rasa.nlu.training_data.message.Message object at 0x0000020798C395C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C39848>, <rasa.nlu.training_data.message.Message object at 0x0000020798C39AC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C39D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C39CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C3D1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C3D648>, <rasa.nlu.training_data.message.Message object at 0x0000020798C3D948>, <rasa.nlu.training_data.message.Message object at 0x0000020798C3DC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C3DF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C44108>, <rasa.nlu.training_data.message.Message object at 0x0000020798C44588>, <rasa.nlu.training_data.message.Message object at 0x0000020798C44888>, <rasa.nlu.training_data.message.Message object at 0x0000020798C44B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C44E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C44F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4B4C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4B7C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4BAC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4BDC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4BD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4F1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4F588>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4F808>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4FA88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4FD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C4FC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C55108>, <rasa.nlu.training_data.message.Message object at 0x0000020798C554C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C55748>, <rasa.nlu.training_data.message.Message object at 0x0000020798C559C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C55C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C55EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C55F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5B408>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5B688>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5B908>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5BB88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5BE08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5BD88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C63288>, <rasa.nlu.training_data.message.Message object at 0x0000020798C63208>, <rasa.nlu.training_data.message.Message object at 0x0000020798C63488>, <rasa.nlu.training_data.message.Message object at 0x0000020798C63708>, <rasa.nlu.training_data.message.Message object at 0x0000020798C63908>, <rasa.nlu.training_data.message.Message object at 0x0000020798C63A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C63C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C63D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6A088>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6A0C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6A248>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6A3C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6A548>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6A6C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6A848>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6A9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6AB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6ACC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C6AF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70208>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70188>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70308>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70488>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70608>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70788>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70908>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C70D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C79088>, <rasa.nlu.training_data.message.Message object at 0x0000020798C790C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C79248>, <rasa.nlu.training_data.message.Message object at 0x0000020798C793C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C79548>, <rasa.nlu.training_data.message.Message object at 0x0000020798C796C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C79848>, <rasa.nlu.training_data.message.Message object at 0x0000020798C799C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C79B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C79CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C79F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85208>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85188>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85308>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85488>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85608>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85788>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85908>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C85F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C8B088>, <rasa.nlu.training_data.message.Message object at 0x0000020798C8B288>, <rasa.nlu.training_data.message.Message object at 0x0000020798C8B488>, <rasa.nlu.training_data.message.Message object at 0x0000020798C8B688>, <rasa.nlu.training_data.message.Message object at 0x0000020798C8B888>, <rasa.nlu.training_data.message.Message object at 0x0000020798C8BA88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C8BC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C90048>, <rasa.nlu.training_data.message.Message object at 0x0000020798C900C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C90288>, <rasa.nlu.training_data.message.Message object at 0x0000020798C90408>, <rasa.nlu.training_data.message.Message object at 0x0000020798C90608>, <rasa.nlu.training_data.message.Message object at 0x0000020798C90808>, <rasa.nlu.training_data.message.Message object at 0x0000020798C90A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C90C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C90F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C94048>, <rasa.nlu.training_data.message.Message object at 0x0000020798C94248>, <rasa.nlu.training_data.message.Message object at 0x0000020798C94448>, <rasa.nlu.training_data.message.Message object at 0x0000020798C94648>, <rasa.nlu.training_data.message.Message object at 0x0000020798C94848>, <rasa.nlu.training_data.message.Message object at 0x0000020798C94A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C94C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C94F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C99088>, <rasa.nlu.training_data.message.Message object at 0x0000020798C99288>, <rasa.nlu.training_data.message.Message object at 0x0000020798C99488>, <rasa.nlu.training_data.message.Message object at 0x0000020798C99688>, <rasa.nlu.training_data.message.Message object at 0x0000020798C99888>, <rasa.nlu.training_data.message.Message object at 0x0000020798C99A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C99C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDA048>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDA088>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDA288>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDA488>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDA988>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDAB08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDAC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDAE08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDAF88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BDAFC8>, <rasa.nlu.training_data.message.Message object at 0x000002073CD69BC8>, <rasa.nlu.training_data.message.Message object at 0x000002073CDB7608>, <rasa.nlu.training_data.message.Message object at 0x000002073CDB7588>, <rasa.nlu.training_data.message.Message object at 0x0000020798EDDB08>, <rasa.nlu.training_data.message.Message object at 0x0000020798EDDBC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EDD688>, <rasa.nlu.training_data.message.Message object at 0x0000020798EDD388>, <rasa.nlu.training_data.message.Message object at 0x0000020798EDD088>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED7D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED7A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED7748>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED7448>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED7088>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED0C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED0848>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED0448>, <rasa.nlu.training_data.message.Message object at 0x0000020798ED0048>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7D2C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7D448>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7D5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7D748>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7D6C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7D848>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7D9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7DB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7DD88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C7DEC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B68088>, <rasa.nlu.training_data.message.Message object at 0x0000020798B68408>, <rasa.nlu.training_data.message.Message object at 0x0000020798B68948>, <rasa.nlu.training_data.message.Message object at 0x0000020798B68C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B68708>, <rasa.nlu.training_data.message.Message object at 0x0000020798B213C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B21648>, <rasa.nlu.training_data.message.Message object at 0x0000020798B218C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B21CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B21F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B21F48>, <rasa.nlu.training_data.message.Message object at 0x000002073CD69CC8>, <rasa.nlu.training_data.message.Message object at 0x000002073CDBF088>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADA208>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADA408>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADA708>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADAA08>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADAC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798ADA908>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAA3C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAA5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAA7C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAA9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAAC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAAE08>, <rasa.nlu.training_data.message.Message object at 0x0000020798AAAE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798AB6188>, <rasa.nlu.training_data.message.Message object at 0x0000020798AB6488>, <rasa.nlu.training_data.message.Message object at 0x0000020798AB6688>, <rasa.nlu.training_data.message.Message object at 0x0000020798AB68C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AB6CC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AB6EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AB6E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798AEB088>, <rasa.nlu.training_data.message.Message object at 0x0000020798AEB0C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AEB448>, <rasa.nlu.training_data.message.Message object at 0x0000020798AEB688>, <rasa.nlu.training_data.message.Message object at 0x0000020798AEB9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AEBDC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798AEB1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B41048>, <rasa.nlu.training_data.message.Message object at 0x0000020798B412C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B41548>, <rasa.nlu.training_data.message.Message object at 0x0000020798B41948>, <rasa.nlu.training_data.message.Message object at 0x0000020798B41B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B41EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B62248>, <rasa.nlu.training_data.message.Message object at 0x0000020798B62648>, <rasa.nlu.training_data.message.Message object at 0x0000020798B62408>, <rasa.nlu.training_data.message.Message object at 0x0000020798B627C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B62A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B6F088>, <rasa.nlu.training_data.message.Message object at 0x0000020798B6F0C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B6F288>, <rasa.nlu.training_data.message.Message object at 0x0000020798B6F548>, <rasa.nlu.training_data.message.Message object at 0x0000020798B6F7C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B6FC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B6FA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B73088>, <rasa.nlu.training_data.message.Message object at 0x0000020798B732C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B73548>, <rasa.nlu.training_data.message.Message object at 0x0000020798B737C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B739C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B73C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9FAC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9F348>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9F548>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9F748>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9F948>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9FB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9FD48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9FE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C9FCC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B7B288>, <rasa.nlu.training_data.message.Message object at 0x0000020798B7B508>, <rasa.nlu.training_data.message.Message object at 0x0000020798B7B788>, <rasa.nlu.training_data.message.Message object at 0x0000020798B7BA88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B7BE08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B7B148>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEE148>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEE388>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEE508>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEE688>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEE808>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEE988>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEEB08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEEC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEEE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798BEECC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF4088>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF40C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF4248>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF43C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF4548>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF46C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF4848>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF49C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF4C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF4D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798BF4A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C181C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C18148>, <rasa.nlu.training_data.message.Message object at 0x0000020798C18588>, <rasa.nlu.training_data.message.Message object at 0x0000020798C18888>, <rasa.nlu.training_data.message.Message object at 0x0000020798C18A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C18B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C18D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C18E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C18F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C185C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C34188>, <rasa.nlu.training_data.message.Message object at 0x0000020798C34548>, <rasa.nlu.training_data.message.Message object at 0x0000020798C347C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C34B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C34E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798C34D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C34A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5F048>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5F2C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5F548>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5F7C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5FB88>, <rasa.nlu.training_data.message.Message object at 0x0000020798C5F908>, <rasa.nlu.training_data.message.Message object at 0x0000020798958D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798958D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798958C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798978508>, <rasa.nlu.training_data.message.Message object at 0x0000020798978208>, <rasa.nlu.training_data.message.Message object at 0x0000020798978048>, <rasa.nlu.training_data.message.Message object at 0x000002079870FF88>, <rasa.nlu.training_data.message.Message object at 0x00000207989A6E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B8F748>, <rasa.nlu.training_data.message.Message object at 0x0000020798B8F848>, <rasa.nlu.training_data.message.Message object at 0x0000020798B8F9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B8F8C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B8FA88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B8F1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B8F388>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4088>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4288>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA43C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4508>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4608>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4788>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4888>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4988>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CA4F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B85D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B85808>, <rasa.nlu.training_data.message.Message object at 0x0000020798B85308>, <rasa.nlu.training_data.message.Message object at 0x0000020798B85248>, <rasa.nlu.training_data.message.Message object at 0x0000020798B85748>, <rasa.nlu.training_data.message.Message object at 0x0000020798B85C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798B84D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798B84888>, <rasa.nlu.training_data.message.Message object at 0x0000020798B84388>, <rasa.nlu.training_data.message.Message object at 0x0000020798B84188>, <rasa.nlu.training_data.message.Message object at 0x0000020798B84688>, <rasa.nlu.training_data.message.Message object at 0x0000020798B84B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB0F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB0A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB0548>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB0048>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB0488>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB0988>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB0E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CAED48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CAE848>, <rasa.nlu.training_data.message.Message object at 0x0000020798CAE348>, <rasa.nlu.training_data.message.Message object at 0x0000020798CAE288>, <rasa.nlu.training_data.message.Message object at 0x0000020798CAE788>, <rasa.nlu.training_data.message.Message object at 0x0000020798CAEC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB3DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB38C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB33C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB31C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB36C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB3BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB6F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB6908>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB6308>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB6248>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB68C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CB6EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CBAB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CBA548>, <rasa.nlu.training_data.message.Message object at 0x0000020798CBA088>, <rasa.nlu.training_data.message.Message object at 0x0000020798CBA688>, <rasa.nlu.training_data.message.Message object at 0x0000020798CBAC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798959D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798959788>, <rasa.nlu.training_data.message.Message object at 0x0000020798959188>, <rasa.nlu.training_data.message.Message object at 0x00000207989595C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798959BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798960E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798960848>, <rasa.nlu.training_data.message.Message object at 0x0000020798960248>, <rasa.nlu.training_data.message.Message object at 0x0000020798960388>, <rasa.nlu.training_data.message.Message object at 0x0000020798960988>, <rasa.nlu.training_data.message.Message object at 0x0000020798960F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798967A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798967488>, <rasa.nlu.training_data.message.Message object at 0x00000207989672C8>, <rasa.nlu.training_data.message.Message object at 0x00000207989678C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798967EC8>, <rasa.nlu.training_data.message.Message object at 0x000002079896DB48>, <rasa.nlu.training_data.message.Message object at 0x000002079896D548>, <rasa.nlu.training_data.message.Message object at 0x000002079896D088>, <rasa.nlu.training_data.message.Message object at 0x000002079896D688>, <rasa.nlu.training_data.message.Message object at 0x000002079896DC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798972D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798972788>, <rasa.nlu.training_data.message.Message object at 0x0000020798972188>, <rasa.nlu.training_data.message.Message object at 0x00000207989725C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798972BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC1E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC1848>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC1248>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC1388>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC1988>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC1F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC6A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC6488>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC62C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC68C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CC6EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CCCB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CCC548>, <rasa.nlu.training_data.message.Message object at 0x0000020798CCC088>, <rasa.nlu.training_data.message.Message object at 0x0000020798CCC688>, <rasa.nlu.training_data.message.Message object at 0x0000020798CCCC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD1D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD1788>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD1188>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD15C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD1BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD6E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD6848>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD6248>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD6388>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD6988>, <rasa.nlu.training_data.message.Message object at 0x0000020798CD6F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CDDA88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CDD488>, <rasa.nlu.training_data.message.Message object at 0x0000020798CDD2C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CDD8C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CDDEC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE2B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE2548>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE2088>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE2688>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE2C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE7D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE7788>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE7188>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE75C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CE7BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CEDE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CED848>, <rasa.nlu.training_data.message.Message object at 0x0000020798CED248>, <rasa.nlu.training_data.message.Message object at 0x0000020798CED388>, <rasa.nlu.training_data.message.Message object at 0x0000020798CED988>, <rasa.nlu.training_data.message.Message object at 0x0000020798CEDF88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF2A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF2488>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF22C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF28C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF2F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF8A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF8408>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF8248>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF8848>, <rasa.nlu.training_data.message.Message object at 0x0000020798CF8E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CFEC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798CFE648>, <rasa.nlu.training_data.message.Message object at 0x0000020798CFE048>, <rasa.nlu.training_data.message.Message object at 0x0000020798CFE608>, <rasa.nlu.training_data.message.Message object at 0x0000020798CFEC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D04D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D04708>, <rasa.nlu.training_data.message.Message object at 0x0000020798D04108>, <rasa.nlu.training_data.message.Message object at 0x0000020798D04548>, <rasa.nlu.training_data.message.Message object at 0x0000020798D04B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D0AF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D0A948>, <rasa.nlu.training_data.message.Message object at 0x0000020798D0A348>, <rasa.nlu.training_data.message.Message object at 0x0000020798D0A308>, <rasa.nlu.training_data.message.Message object at 0x0000020798D0A908>, <rasa.nlu.training_data.message.Message object at 0x0000020798D0AF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D10A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D10408>, <rasa.nlu.training_data.message.Message object at 0x0000020798D10248>, <rasa.nlu.training_data.message.Message object at 0x0000020798D10848>, <rasa.nlu.training_data.message.Message object at 0x0000020798D10E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D16C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D16648>, <rasa.nlu.training_data.message.Message object at 0x0000020798D16048>, <rasa.nlu.training_data.message.Message object at 0x0000020798D16608>, <rasa.nlu.training_data.message.Message object at 0x0000020798D16C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D1BD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D1B708>, <rasa.nlu.training_data.message.Message object at 0x0000020798D1B108>, <rasa.nlu.training_data.message.Message object at 0x0000020798D1B548>, <rasa.nlu.training_data.message.Message object at 0x0000020798D1BB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D22F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D22948>, <rasa.nlu.training_data.message.Message object at 0x0000020798D22348>, <rasa.nlu.training_data.message.Message object at 0x0000020798D22308>, <rasa.nlu.training_data.message.Message object at 0x0000020798D22908>, <rasa.nlu.training_data.message.Message object at 0x0000020798D22F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D27A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D27408>, <rasa.nlu.training_data.message.Message object at 0x0000020798D27248>, <rasa.nlu.training_data.message.Message object at 0x0000020798D27848>, <rasa.nlu.training_data.message.Message object at 0x0000020798D27E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D2DC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D2D648>, <rasa.nlu.training_data.message.Message object at 0x0000020798D2D048>, <rasa.nlu.training_data.message.Message object at 0x0000020798D2D608>, <rasa.nlu.training_data.message.Message object at 0x0000020798D2DC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D32D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D32708>, <rasa.nlu.training_data.message.Message object at 0x0000020798D32108>, <rasa.nlu.training_data.message.Message object at 0x0000020798D32548>, <rasa.nlu.training_data.message.Message object at 0x0000020798D32B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D36F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D36948>, <rasa.nlu.training_data.message.Message object at 0x0000020798D36348>, <rasa.nlu.training_data.message.Message object at 0x0000020798D36308>, <rasa.nlu.training_data.message.Message object at 0x0000020798D36908>, <rasa.nlu.training_data.message.Message object at 0x0000020798D36F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D3DA08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D3D408>, <rasa.nlu.training_data.message.Message object at 0x0000020798D3D248>, <rasa.nlu.training_data.message.Message object at 0x0000020798D3D848>, <rasa.nlu.training_data.message.Message object at 0x0000020798D3DE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D44C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D44648>, <rasa.nlu.training_data.message.Message object at 0x0000020798D44048>, <rasa.nlu.training_data.message.Message object at 0x0000020798D44608>, <rasa.nlu.training_data.message.Message object at 0x0000020798D44C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D4BD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D4B708>, <rasa.nlu.training_data.message.Message object at 0x0000020798D4B108>, <rasa.nlu.training_data.message.Message object at 0x0000020798D4B548>, <rasa.nlu.training_data.message.Message object at 0x0000020798D4BB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D50F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D50948>, <rasa.nlu.training_data.message.Message object at 0x0000020798D50348>, <rasa.nlu.training_data.message.Message object at 0x0000020798D50308>, <rasa.nlu.training_data.message.Message object at 0x0000020798D50908>, <rasa.nlu.training_data.message.Message object at 0x0000020798D50F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D56A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D56408>, <rasa.nlu.training_data.message.Message object at 0x0000020798D56248>, <rasa.nlu.training_data.message.Message object at 0x0000020798D56848>, <rasa.nlu.training_data.message.Message object at 0x0000020798D56E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D5BC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D5B648>, <rasa.nlu.training_data.message.Message object at 0x0000020798D5B048>, <rasa.nlu.training_data.message.Message object at 0x0000020798D5B608>, <rasa.nlu.training_data.message.Message object at 0x0000020798D5BC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D61D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D61708>, <rasa.nlu.training_data.message.Message object at 0x0000020798D61108>, <rasa.nlu.training_data.message.Message object at 0x0000020798D61548>, <rasa.nlu.training_data.message.Message object at 0x0000020798D61B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D67F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D67948>, <rasa.nlu.training_data.message.Message object at 0x0000020798D67348>, <rasa.nlu.training_data.message.Message object at 0x0000020798D67308>, <rasa.nlu.training_data.message.Message object at 0x0000020798D67908>, <rasa.nlu.training_data.message.Message object at 0x0000020798D67F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D6EA08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D6E408>, <rasa.nlu.training_data.message.Message object at 0x0000020798D6E248>, <rasa.nlu.training_data.message.Message object at 0x0000020798D6E848>, <rasa.nlu.training_data.message.Message object at 0x0000020798D6EE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D74C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D74648>, <rasa.nlu.training_data.message.Message object at 0x0000020798D74048>, <rasa.nlu.training_data.message.Message object at 0x0000020798D74608>, <rasa.nlu.training_data.message.Message object at 0x0000020798D74C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7BD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7B708>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7B108>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7B548>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7BB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7FF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7F948>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7F348>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7F308>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7F908>, <rasa.nlu.training_data.message.Message object at 0x0000020798D7FF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D85A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D85408>, <rasa.nlu.training_data.message.Message object at 0x0000020798D85248>, <rasa.nlu.training_data.message.Message object at 0x0000020798D85848>, <rasa.nlu.training_data.message.Message object at 0x0000020798D85E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798D8AB08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D8A508>, <rasa.nlu.training_data.message.Message object at 0x0000020798D8A048>, <rasa.nlu.training_data.message.Message object at 0x0000020798D8A648>, <rasa.nlu.training_data.message.Message object at 0x0000020798D8AC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D90D48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D90748>, <rasa.nlu.training_data.message.Message object at 0x0000020798D90148>, <rasa.nlu.training_data.message.Message object at 0x0000020798D90588>, <rasa.nlu.training_data.message.Message object at 0x0000020798D90B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798D94E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798D94808>, <rasa.nlu.training_data.message.Message object at 0x0000020798D94208>, <rasa.nlu.training_data.message.Message object at 0x0000020798D94348>, <rasa.nlu.training_data.message.Message object at 0x0000020798D94948>, <rasa.nlu.training_data.message.Message object at 0x0000020798D94F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D9BA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798D9B448>, <rasa.nlu.training_data.message.Message object at 0x0000020798D9B288>, <rasa.nlu.training_data.message.Message object at 0x0000020798D9B888>, <rasa.nlu.training_data.message.Message object at 0x0000020798D9BE88>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA2B08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA2508>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA2048>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA2648>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA2C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA8D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA8788>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA8148>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA85C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798DA8BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798DAEE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DAE848>, <rasa.nlu.training_data.message.Message object at 0x0000020798DAE248>, <rasa.nlu.training_data.message.Message object at 0x0000020798DAE388>, <rasa.nlu.training_data.message.Message object at 0x0000020798DAE988>, <rasa.nlu.training_data.message.Message object at 0x0000020798DAEF88>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB4A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB4488>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB42C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB48C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB4EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB9B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB9548>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB9088>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB9688>, <rasa.nlu.training_data.message.Message object at 0x0000020798DB9C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798DBFD88>, <rasa.nlu.training_data.message.Message object at 0x0000020798DBF788>, <rasa.nlu.training_data.message.Message object at 0x0000020798DBF188>, <rasa.nlu.training_data.message.Message object at 0x0000020798DBF5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798DBFBC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798DC4D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DC4708>, <rasa.nlu.training_data.message.Message object at 0x0000020798DC4108>, <rasa.nlu.training_data.message.Message object at 0x0000020798DC4548>, <rasa.nlu.training_data.message.Message object at 0x0000020798DC4B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCBF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCB948>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCB348>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCB308>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCB908>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCBF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCFA08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCF408>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCF248>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCF848>, <rasa.nlu.training_data.message.Message object at 0x0000020798DCFE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD5C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD5648>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD5048>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD5608>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD5C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD9D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD9708>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD9108>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD9548>, <rasa.nlu.training_data.message.Message object at 0x0000020798DD9B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DDFF48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DDF948>, <rasa.nlu.training_data.message.Message object at 0x0000020798DDF348>, <rasa.nlu.training_data.message.Message object at 0x0000020798DDF308>, <rasa.nlu.training_data.message.Message object at 0x0000020798DDF908>, <rasa.nlu.training_data.message.Message object at 0x0000020798DDFF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE4A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE4408>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE4248>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE4848>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE4E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE9C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE9648>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE9048>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE9608>, <rasa.nlu.training_data.message.Message object at 0x0000020798DE9C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DEED08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DEE708>, <rasa.nlu.training_data.message.Message object at 0x0000020798DEE108>, <rasa.nlu.training_data.message.Message object at 0x0000020798DEE548>, <rasa.nlu.training_data.message.Message object at 0x0000020798DEEB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF2F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF2948>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF2348>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF2308>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF2908>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF2F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF8A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF8408>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF8248>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF8848>, <rasa.nlu.training_data.message.Message object at 0x0000020798DF8E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFCC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFC648>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFC048>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFC608>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFCC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFFD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFF708>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFF108>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFF548>, <rasa.nlu.training_data.message.Message object at 0x0000020798DFFB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E05F48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E05948>, <rasa.nlu.training_data.message.Message object at 0x0000020798E05348>, <rasa.nlu.training_data.message.Message object at 0x0000020798E05308>, <rasa.nlu.training_data.message.Message object at 0x0000020798E05908>, <rasa.nlu.training_data.message.Message object at 0x0000020798E05F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E09A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E09408>, <rasa.nlu.training_data.message.Message object at 0x0000020798E09248>, <rasa.nlu.training_data.message.Message object at 0x0000020798E09848>, <rasa.nlu.training_data.message.Message object at 0x0000020798E09E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E0DC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E0D648>, <rasa.nlu.training_data.message.Message object at 0x0000020798E0D048>, <rasa.nlu.training_data.message.Message object at 0x0000020798E0D608>, <rasa.nlu.training_data.message.Message object at 0x0000020798E0DC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E12D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E12708>, <rasa.nlu.training_data.message.Message object at 0x0000020798E12108>, <rasa.nlu.training_data.message.Message object at 0x0000020798E12548>, <rasa.nlu.training_data.message.Message object at 0x0000020798E12B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E17988>, <rasa.nlu.training_data.message.Message object at 0x0000020798E17C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E177C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E171C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E17488>, <rasa.nlu.training_data.message.Message object at 0x0000020798E17A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E17E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E1DDC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E1D9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E1D5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E1D1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E1D288>, <rasa.nlu.training_data.message.Message object at 0x0000020798E1D688>, <rasa.nlu.training_data.message.Message object at 0x0000020798E1DA88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E1DE88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E24D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E24908>, <rasa.nlu.training_data.message.Message object at 0x0000020798E24508>, <rasa.nlu.training_data.message.Message object at 0x0000020798E24108>, <rasa.nlu.training_data.message.Message object at 0x0000020798E243C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E247C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E24BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E24FC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E29C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E29848>, <rasa.nlu.training_data.message.Message object at 0x0000020798E29448>, <rasa.nlu.training_data.message.Message object at 0x0000020798E29048>, <rasa.nlu.training_data.message.Message object at 0x0000020798E29408>, <rasa.nlu.training_data.message.Message object at 0x0000020798E29808>, <rasa.nlu.training_data.message.Message object at 0x0000020798E29C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E2FF88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E2FB88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E2F788>, <rasa.nlu.training_data.message.Message object at 0x0000020798E2F388>, <rasa.nlu.training_data.message.Message object at 0x0000020798E2F048>, <rasa.nlu.training_data.message.Message object at 0x0000020798E2F448>, <rasa.nlu.training_data.message.Message object at 0x0000020798E2F848>, <rasa.nlu.training_data.message.Message object at 0x0000020798E2FC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E35FC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E35BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E357C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E353C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E35088>, <rasa.nlu.training_data.message.Message object at 0x0000020798E35488>, <rasa.nlu.training_data.message.Message object at 0x0000020798E35888>, <rasa.nlu.training_data.message.Message object at 0x0000020798E35C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E3DF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E3DB08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E3D708>, <rasa.nlu.training_data.message.Message object at 0x0000020798E3D308>, <rasa.nlu.training_data.message.Message object at 0x0000020798E3D1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E3D5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E3D9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E3DDC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E44E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E44A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E44648>, <rasa.nlu.training_data.message.Message object at 0x0000020798E44248>, <rasa.nlu.training_data.message.Message object at 0x0000020798E44208>, <rasa.nlu.training_data.message.Message object at 0x0000020798E44608>, <rasa.nlu.training_data.message.Message object at 0x0000020798E44A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E44E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E4CD88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E4C988>, <rasa.nlu.training_data.message.Message object at 0x0000020798E4C588>, <rasa.nlu.training_data.message.Message object at 0x0000020798E4C188>, <rasa.nlu.training_data.message.Message object at 0x0000020798E4C248>, <rasa.nlu.training_data.message.Message object at 0x0000020798E4C648>, <rasa.nlu.training_data.message.Message object at 0x0000020798E4CA48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E4CE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E54DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E549C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E545C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E541C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E54288>, <rasa.nlu.training_data.message.Message object at 0x0000020798E54688>, <rasa.nlu.training_data.message.Message object at 0x0000020798E54A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E54E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E5BD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E5B908>, <rasa.nlu.training_data.message.Message object at 0x0000020798E5B508>, <rasa.nlu.training_data.message.Message object at 0x0000020798E5B108>, <rasa.nlu.training_data.message.Message object at 0x0000020798E5B3C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E5B7C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E5BBC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E5BFC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E63C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E63848>, <rasa.nlu.training_data.message.Message object at 0x0000020798E63448>, <rasa.nlu.training_data.message.Message object at 0x0000020798E63048>, <rasa.nlu.training_data.message.Message object at 0x0000020798E63408>, <rasa.nlu.training_data.message.Message object at 0x0000020798E63808>, <rasa.nlu.training_data.message.Message object at 0x0000020798E63C08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E6CF88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E6CB88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E6C788>, <rasa.nlu.training_data.message.Message object at 0x0000020798E6C388>, <rasa.nlu.training_data.message.Message object at 0x0000020798E6C048>, <rasa.nlu.training_data.message.Message object at 0x0000020798E6C448>, <rasa.nlu.training_data.message.Message object at 0x0000020798E6C848>, <rasa.nlu.training_data.message.Message object at 0x0000020798E6CC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E74FC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E74BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E747C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E743C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E74088>, <rasa.nlu.training_data.message.Message object at 0x0000020798E74488>, <rasa.nlu.training_data.message.Message object at 0x0000020798E74888>, <rasa.nlu.training_data.message.Message object at 0x0000020798E74C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E7BF08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E7BB08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E7B708>, <rasa.nlu.training_data.message.Message object at 0x0000020798E7B308>, <rasa.nlu.training_data.message.Message object at 0x0000020798E7B1C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E7B5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E7B9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E7BDC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E83E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E83A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E83648>, <rasa.nlu.training_data.message.Message object at 0x0000020798E83248>, <rasa.nlu.training_data.message.Message object at 0x0000020798E83208>, <rasa.nlu.training_data.message.Message object at 0x0000020798E83608>, <rasa.nlu.training_data.message.Message object at 0x0000020798E83A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E83E08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E89D88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E89988>, <rasa.nlu.training_data.message.Message object at 0x0000020798E89588>, <rasa.nlu.training_data.message.Message object at 0x0000020798E89188>, <rasa.nlu.training_data.message.Message object at 0x0000020798E89248>, <rasa.nlu.training_data.message.Message object at 0x0000020798E89648>, <rasa.nlu.training_data.message.Message object at 0x0000020798E89A48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E89E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E92DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E929C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E925C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E921C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E92288>, <rasa.nlu.training_data.message.Message object at 0x0000020798E92688>, <rasa.nlu.training_data.message.Message object at 0x0000020798E92A88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E92E88>, <rasa.nlu.training_data.message.Message object at 0x0000020798E98D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798E98908>, <rasa.nlu.training_data.message.Message object at 0x0000020798E98508>, <rasa.nlu.training_data.message.Message object at 0x0000020798E98108>, <rasa.nlu.training_data.message.Message object at 0x0000020798E983C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E987C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E98BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E98FC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798E9DC48>, <rasa.nlu.training_data.message.Message object at 0x0000020798E9D848>, <rasa.nlu.training_data.message.Message object at 0x0000020798E9D448>, <rasa.nlu.training_data.message.Message object at 0x0000020798E9D048>, <rasa.nlu.training_data.message.Message object at 0x0000020798E9D408>, <rasa.nlu.training_data.message.Message object at 0x0000020798E9D808>, <rasa.nlu.training_data.message.Message object at 0x0000020798E9DC08>, <rasa.nlu.training_data.message.Message object at 0x0000020798EA4F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798EA4B88>, <rasa.nlu.training_data.message.Message object at 0x0000020798EA4788>, <rasa.nlu.training_data.message.Message object at 0x0000020798EA4388>, <rasa.nlu.training_data.message.Message object at 0x0000020798EA4048>, <rasa.nlu.training_data.message.Message object at 0x0000020798EA4448>, <rasa.nlu.training_data.message.Message object at 0x0000020798EA4848>, <rasa.nlu.training_data.message.Message object at 0x0000020798EA4C48>, <rasa.nlu.training_data.message.Message object at 0x0000020798EAAFC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EAABC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EAA7C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EAA3C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EAA088>, <rasa.nlu.training_data.message.Message object at 0x0000020798EAA488>, <rasa.nlu.training_data.message.Message object at 0x0000020798EAA888>, <rasa.nlu.training_data.message.Message object at 0x0000020798EAAC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB1F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB1B08>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB1708>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB1308>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB11C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB15C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB18C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB1BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB1EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7E48>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7B48>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7848>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7548>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7248>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7108>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7408>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7708>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7A08>, <rasa.nlu.training_data.message.Message object at 0x0000020798EB7D08>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBEF88>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBEC88>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBE988>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBE688>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBE388>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBE088>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBE248>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBE548>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBE848>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBEB48>, <rasa.nlu.training_data.message.Message object at 0x0000020798EBEE48>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC4EC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC4BC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC48C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC45C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC42C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC4088>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC4388>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC4688>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC4988>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC4C88>, <rasa.nlu.training_data.message.Message object at 0x0000020798EC4F88>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECAD08>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECAA08>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECA708>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECA408>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECA108>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECA2C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECA5C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECA8C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798ECABC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A8C0C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A8C508>, <rasa.nlu.training_data.message.Message object at 0x0000020798A8C248>, <rasa.nlu.training_data.message.Message object at 0x0000020798A8C9C8>, <rasa.nlu.training_data.message.Message object at 0x0000020798A8C708>, <rasa.nlu.training_data.message.Message object at 0x000002073895E7C8>, <rasa.nlu.training_data.message.Message object at 0x00000207361A1DC8>, <rasa.nlu.training_data.message.Message object at 0x00000207361A1188>, <rasa.nlu.training_data.message.Message object at 0x00000207361A1248>, <rasa.nlu.training_data.message.Message object at 0x00000207361A1C88>, <rasa.nlu.training_data.message.Message object at 0x000002077E5CA148>, <rasa.nlu.training_data.message.Message object at 0x0000020798B90F08>, <rasa.nlu.training_data.message.Message object at 0x0000020798B90DC8>, <rasa.nlu.training_data.message.Message object at 0x0000020798B90C88>], 'entity_synonyms': {}, 'regex_features': [], 'lookup_tables': [], 'nlg_stories': {}}
    

    WARNING:root:Sequence length will auto set at 95% of sequence length
    WARNING:root:seq_len: 19
    

    Model: "model_4"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Input-Token (InputLayer)        [(None, 19)]         0                                            
    __________________________________________________________________________________________________
    Input-Segment (InputLayer)      [(None, 19)]         0                                            
    __________________________________________________________________________________________________
    Embedding-Token (TokenEmbedding [(None, 19, 768), (2 16226304    Input-Token[0][0]                
    __________________________________________________________________________________________________
    Embedding-Segment (Embedding)   (None, 19, 768)      1536        Input-Segment[0][0]              
    __________________________________________________________________________________________________
    Embedding-Token-Segment (Add)   (None, 19, 768)      0           Embedding-Token[0][0]            
                                                                     Embedding-Segment[0][0]          
    __________________________________________________________________________________________________
    Embedding-Position (PositionEmb (None, 19, 768)      14592       Embedding-Token-Segment[0][0]    
    __________________________________________________________________________________________________
    Embedding-Dropout (Dropout)     (None, 19, 768)      0           Embedding-Position[0][0]         
    __________________________________________________________________________________________________
    Embedding-Norm (LayerNormalizat (None, 19, 768)      1536        Embedding-Dropout[0][0]          
    __________________________________________________________________________________________________
    Encoder-1-MultiHeadSelfAttentio (None, 19, 768)      2362368     Embedding-Norm[0][0]             
    __________________________________________________________________________________________________
    Encoder-1-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-1-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-1-MultiHeadSelfAttentio (None, 19, 768)      0           Embedding-Norm[0][0]             
                                                                     Encoder-1-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-1-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-1-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-1-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-1-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-1-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-1-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-1-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-1-MultiHeadSelfAttention-
                                                                     Encoder-1-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-1-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-1-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-2-MultiHeadSelfAttentio (None, 19, 768)      2362368     Encoder-1-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-2-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-2-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-2-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-1-FeedForward-Norm[0][0] 
                                                                     Encoder-2-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-2-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-2-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-2-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-2-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-2-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-2-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-2-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-2-MultiHeadSelfAttention-
                                                                     Encoder-2-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-2-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-2-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-3-MultiHeadSelfAttentio (None, 19, 768)      2362368     Encoder-2-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-3-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-3-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-3-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-2-FeedForward-Norm[0][0] 
                                                                     Encoder-3-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-3-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-3-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-3-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-3-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-3-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-3-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-3-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-3-MultiHeadSelfAttention-
                                                                     Encoder-3-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-3-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-3-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-4-MultiHeadSelfAttentio (None, 19, 768)      2362368     Encoder-3-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-4-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-4-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-4-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-3-FeedForward-Norm[0][0] 
                                                                     Encoder-4-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-4-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-4-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-4-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-4-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-4-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-4-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-4-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-4-MultiHeadSelfAttention-
                                                                     Encoder-4-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-4-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-4-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-5-MultiHeadSelfAttentio (None, 19, 768)      2362368     Encoder-4-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-5-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-5-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-5-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-4-FeedForward-Norm[0][0] 
                                                                     Encoder-5-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-5-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-5-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-5-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-5-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-5-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-5-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-5-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-5-MultiHeadSelfAttention-
                                                                     Encoder-5-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-5-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-5-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-6-MultiHeadSelfAttentio (None, 19, 768)      2362368     Encoder-5-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-6-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-6-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-6-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-5-FeedForward-Norm[0][0] 
                                                                     Encoder-6-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-6-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-6-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-6-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-6-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-6-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-6-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-6-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-6-MultiHeadSelfAttention-
                                                                     Encoder-6-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-6-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-6-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-7-MultiHeadSelfAttentio (None, 19, 768)      2362368     Encoder-6-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-7-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-7-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-7-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-6-FeedForward-Norm[0][0] 
                                                                     Encoder-7-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-7-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-7-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-7-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-7-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-7-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-7-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-7-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-7-MultiHeadSelfAttention-
                                                                     Encoder-7-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-7-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-7-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-8-MultiHeadSelfAttentio (None, 19, 768)      2362368     Encoder-7-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-8-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-8-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-8-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-7-FeedForward-Norm[0][0] 
                                                                     Encoder-8-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-8-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-8-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-8-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-8-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-8-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-8-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-8-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-8-MultiHeadSelfAttention-
                                                                     Encoder-8-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-8-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-8-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-9-MultiHeadSelfAttentio (None, 19, 768)      2362368     Encoder-8-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-9-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-9-MultiHeadSelfAttention[
    __________________________________________________________________________________________________
    Encoder-9-MultiHeadSelfAttentio (None, 19, 768)      0           Encoder-8-FeedForward-Norm[0][0] 
                                                                     Encoder-9-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-9-MultiHeadSelfAttentio (None, 19, 768)      1536        Encoder-9-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-9-FeedForward (FeedForw (None, 19, 768)      4722432     Encoder-9-MultiHeadSelfAttention-
    __________________________________________________________________________________________________
    Encoder-9-FeedForward-Dropout ( (None, 19, 768)      0           Encoder-9-FeedForward[0][0]      
    __________________________________________________________________________________________________
    Encoder-9-FeedForward-Add (Add) (None, 19, 768)      0           Encoder-9-MultiHeadSelfAttention-
                                                                     Encoder-9-FeedForward-Dropout[0][
    __________________________________________________________________________________________________
    Encoder-9-FeedForward-Norm (Lay (None, 19, 768)      1536        Encoder-9-FeedForward-Add[0][0]  
    __________________________________________________________________________________________________
    Encoder-10-MultiHeadSelfAttenti (None, 19, 768)      2362368     Encoder-9-FeedForward-Norm[0][0] 
    __________________________________________________________________________________________________
    Encoder-10-MultiHeadSelfAttenti (None, 19, 768)      0           Encoder-10-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-10-MultiHeadSelfAttenti (None, 19, 768)      0           Encoder-9-FeedForward-Norm[0][0] 
                                                                     Encoder-10-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-10-MultiHeadSelfAttenti (None, 19, 768)      1536        Encoder-10-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-10-FeedForward (FeedFor (None, 19, 768)      4722432     Encoder-10-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-10-FeedForward-Dropout  (None, 19, 768)      0           Encoder-10-FeedForward[0][0]     
    __________________________________________________________________________________________________
    Encoder-10-FeedForward-Add (Add (None, 19, 768)      0           Encoder-10-MultiHeadSelfAttention
                                                                     Encoder-10-FeedForward-Dropout[0]
    __________________________________________________________________________________________________
    Encoder-10-FeedForward-Norm (La (None, 19, 768)      1536        Encoder-10-FeedForward-Add[0][0] 
    __________________________________________________________________________________________________
    Encoder-11-MultiHeadSelfAttenti (None, 19, 768)      2362368     Encoder-10-FeedForward-Norm[0][0]
    __________________________________________________________________________________________________
    Encoder-11-MultiHeadSelfAttenti (None, 19, 768)      0           Encoder-11-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-11-MultiHeadSelfAttenti (None, 19, 768)      0           Encoder-10-FeedForward-Norm[0][0]
                                                                     Encoder-11-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-11-MultiHeadSelfAttenti (None, 19, 768)      1536        Encoder-11-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-11-FeedForward (FeedFor (None, 19, 768)      4722432     Encoder-11-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-11-FeedForward-Dropout  (None, 19, 768)      0           Encoder-11-FeedForward[0][0]     
    __________________________________________________________________________________________________
    Encoder-11-FeedForward-Add (Add (None, 19, 768)      0           Encoder-11-MultiHeadSelfAttention
                                                                     Encoder-11-FeedForward-Dropout[0]
    __________________________________________________________________________________________________
    Encoder-11-FeedForward-Norm (La (None, 19, 768)      1536        Encoder-11-FeedForward-Add[0][0] 
    __________________________________________________________________________________________________
    Encoder-12-MultiHeadSelfAttenti (None, 19, 768)      2362368     Encoder-11-FeedForward-Norm[0][0]
    __________________________________________________________________________________________________
    Encoder-12-MultiHeadSelfAttenti (None, 19, 768)      0           Encoder-12-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-12-MultiHeadSelfAttenti (None, 19, 768)      0           Encoder-11-FeedForward-Norm[0][0]
                                                                     Encoder-12-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-12-MultiHeadSelfAttenti (None, 19, 768)      1536        Encoder-12-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-12-FeedForward (FeedFor (None, 19, 768)      4722432     Encoder-12-MultiHeadSelfAttention
    __________________________________________________________________________________________________
    Encoder-12-FeedForward-Dropout  (None, 19, 768)      0           Encoder-12-FeedForward[0][0]     
    __________________________________________________________________________________________________
    Encoder-12-FeedForward-Add (Add (None, 19, 768)      0           Encoder-12-MultiHeadSelfAttention
                                                                     Encoder-12-FeedForward-Dropout[0]
    __________________________________________________________________________________________________
    Encoder-12-FeedForward-Norm (La (None, 19, 768)      1536        Encoder-12-FeedForward-Add[0][0] 
    __________________________________________________________________________________________________
    Encoder-Output (Concatenate)    (None, 19, 3072)     0           Encoder-9-FeedForward-Norm[0][0] 
                                                                     Encoder-10-FeedForward-Norm[0][0]
                                                                     Encoder-11-FeedForward-Norm[0][0]
                                                                     Encoder-12-FeedForward-Norm[0][0]
    __________________________________________________________________________________________________
    non_masking_layer (NonMaskingLa (None, 19, 3072)     0           Encoder-Output[0][0]             
    __________________________________________________________________________________________________
    layer_blstm (Bidirectional)     (None, 19, 256)      3277824     non_masking_layer[0][0]          
    __________________________________________________________________________________________________
    layer_dropout (Dropout)         (None, 19, 256)      0           layer_blstm[0][0]                
    __________________________________________________________________________________________________
    layer_time_distributed (TimeDis (None, 19, 28)       7196        layer_dropout[0][0]              
    __________________________________________________________________________________________________
    activation (Activation)         (None, 19, 28)       0           layer_time_distributed[0][0]     
    ==================================================================================================
    Total params: 104,583,452
    Trainable params: 3,285,020
    Non-trainable params: 101,298,432
    __________________________________________________________________________________________________
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    Train for 34 steps, validate for 9 steps
    Epoch 1/10
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
     1/34 [..............................] - ETA: 7:33 - loss: 2.0536 - accuracy: 0.0359调试： (2, 32, 19) (32, 19, 28)
     2/34 [>.............................] - ETA: 3:52 - loss: 1.6021 - accuracy: 0.2511调试： (2, 32, 19) (32, 19, 28)
     3/34 [=>............................] - ETA: 2:38 - loss: 1.4376 - accuracy: 0.3538调试： (2, 32, 19) (32, 19, 28)
     4/34 [==>...........................] - ETA: 2:00 - loss: 1.3595 - accuracy: 0.4127调试： (2, 32, 19) (32, 19, 28)
     5/34 [===>..........................] - ETA: 1:37 - loss: 1.1987 - accuracy: 0.4639调试： (2, 32, 19) (32, 19, 28)
     6/34 [====>.........................] - ETA: 1:22 - loss: 1.1163 - accuracy: 0.5097调试： (2, 32, 19) (32, 19, 28)
     7/34 [=====>........................] - ETA: 1:11 - loss: 1.0587 - accuracy: 0.5357调试： (2, 32, 19) (32, 19, 28)
     8/34 [======>.......................] - ETA: 1:02 - loss: 1.0047 - accuracy: 0.5641调试： (2, 32, 19) (32, 19, 28)
     9/34 [======>.......................] - ETA: 55s - loss: 0.9655 - accuracy: 0.5874 调试： (2, 32, 19) (32, 19, 28)
    10/34 [=======>......................] - ETA: 49s - loss: 0.9282 - accuracy: 0.6052调试： (2, 32, 19) (32, 19, 28)
    11/34 [========>.....................] - ETA: 44s - loss: 0.8912 - accuracy: 0.6181调试： (2, 32, 19) (32, 19, 28)
    12/34 [=========>....................] - ETA: 40s - loss: 0.8562 - accuracy: 0.6300调试： (2, 32, 19) (32, 19, 28)
    13/34 [==========>...................] - ETA: 37s - loss: 0.8188 - accuracy: 0.6435调试： (2, 32, 19) (32, 19, 28)
    14/34 [===========>..................] - ETA: 33s - loss: 0.7916 - accuracy: 0.6580调试： (2, 32, 19) (32, 19, 28)
    15/34 [============>.................] - ETA: 30s - loss: 0.7680 - accuracy: 0.6694调试： (2, 32, 19) (32, 19, 28)
    16/34 [=============>................] - ETA: 28s - loss: 0.7402 - accuracy: 0.6817调试： (2, 32, 19) (32, 19, 28)
    17/34 [==============>...............] - ETA: 25s - loss: 0.7168 - accuracy: 0.6905调试： (2, 32, 19) (32, 19, 28)
    18/34 [==============>...............] - ETA: 23s - loss: 0.6946 - accuracy: 0.7007调试： (2, 32, 19) (32, 19, 28)
    19/34 [===============>..............] - ETA: 21s - loss: 0.6740 - accuracy: 0.7101调试： (2, 32, 19) (32, 19, 28)
    20/34 [================>.............] - ETA: 19s - loss: 0.6528 - accuracy: 0.7183调试： (2, 4, 19) (4, 19, 28)
    21/34 [=================>............] - ETA: 17s - loss: 0.6361 - accuracy: 0.7262调试： (2, 32, 19) (32, 19, 28)
    22/34 [==================>...........] - ETA: 16s - loss: 0.6218 - accuracy: 0.7328调试： (2, 32, 19) (32, 19, 28)
    23/34 [===================>..........] - ETA: 14s - loss: 0.6055 - accuracy: 0.7383调试： (2, 32, 19) (32, 19, 28)
    24/34 [====================>.........] - ETA: 13s - loss: 0.5948 - accuracy: 0.7431调试： (2, 32, 19) (32, 19, 28)
    25/34 [=====================>........] - ETA: 11s - loss: 0.5825 - accuracy: 0.7467调试： (2, 32, 19) (32, 19, 28)
    26/34 [=====================>........] - ETA: 10s - loss: 0.5685 - accuracy: 0.7535调试： (2, 32, 19) (32, 19, 28)
    27/34 [======================>.......] - ETA: 8s - loss: 0.5560 - accuracy: 0.7586 调试： (2, 32, 19) (32, 19, 28)
    28/34 [=======================>......] - ETA: 7s - loss: 0.5412 - accuracy: 0.7647调试： (2, 32, 19) (32, 19, 28)
    29/34 [========================>.....] - ETA: 6s - loss: 0.5315 - accuracy: 0.7697调试： (2, 32, 19) (32, 19, 28)
    30/34 [=========================>....] - ETA: 4s - loss: 0.5233 - accuracy: 0.7725调试： (2, 32, 19) (32, 19, 28)
    31/34 [==========================>...] - ETA: 3s - loss: 0.5119 - accuracy: 0.7775调试： (2, 32, 19) (32, 19, 28)
    32/34 [===========================>..] - ETA: 2s - loss: 0.5017 - accuracy: 0.7815调试： (2, 32, 19) (32, 19, 28)
    33/34 [============================>.] - ETA: 1s - loss: 0.4938 - accuracy: 0.7853调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    
    Epoch 00001: val_loss improved from inf to 0.13893, saving model to entity_weights.h5
    34/34 [==============================] - 52s 2s/step - loss: 0.4808 - accuracy: 0.7861 - val_loss: 0.1389 - val_accuracy: 0.9488
    Epoch 2/10
    调试： (2, 32, 19) (32, 19, 28)
     1/34 [..............................] - ETA: 28s - loss: 0.1619 - accuracy: 0.9385调试： (2, 32, 19) (32, 19, 28)
     2/34 [>.............................] - ETA: 26s - loss: 0.1625 - accuracy: 0.9325调试： (2, 32, 19) (32, 19, 28)
     3/34 [=>............................] - ETA: 25s - loss: 0.1401 - accuracy: 0.9440调试： (2, 32, 19) (32, 19, 28)
     4/34 [==>...........................] - ETA: 24s - loss: 0.1545 - accuracy: 0.9377调试： (2, 32, 19) (32, 19, 28)
     5/34 [===>..........................] - ETA: 22s - loss: 0.1464 - accuracy: 0.9416调试： (2, 32, 19) (32, 19, 28)
     6/34 [====>.........................] - ETA: 21s - loss: 0.1563 - accuracy: 0.9354调试： (2, 32, 19) (32, 19, 28)
     7/34 [=====>........................] - ETA: 21s - loss: 0.1471 - accuracy: 0.9379调试： (2, 32, 19) (32, 19, 28)
     8/34 [======>.......................] - ETA: 20s - loss: 0.1428 - accuracy: 0.9398调试： (2, 32, 19) (32, 19, 28)
     9/34 [======>.......................] - ETA: 19s - loss: 0.1356 - accuracy: 0.9432调试： (2, 32, 19) (32, 19, 28)
    10/34 [=======>......................] - ETA: 18s - loss: 0.1372 - accuracy: 0.9427调试： (2, 32, 19) (32, 19, 28)
    11/34 [========>.....................] - ETA: 18s - loss: 0.1337 - accuracy: 0.9432调试： (2, 32, 19) (32, 19, 28)
    12/34 [=========>....................] - ETA: 17s - loss: 0.1329 - accuracy: 0.9441调试： (2, 32, 19) (32, 19, 28)
    13/34 [==========>...................] - ETA: 16s - loss: 0.1368 - accuracy: 0.9406调试： (2, 32, 19) (32, 19, 28)
    14/34 [===========>..................] - ETA: 15s - loss: 0.1350 - accuracy: 0.9414调试： (2, 32, 19) (32, 19, 28)
    15/34 [============>.................] - ETA: 14s - loss: 0.1328 - accuracy: 0.9417调试： (2, 32, 19) (32, 19, 28)
    16/34 [=============>................] - ETA: 14s - loss: 0.1325 - accuracy: 0.9420调试： (2, 32, 19) (32, 19, 28)
    17/34 [==============>...............] - ETA: 13s - loss: 0.1344 - accuracy: 0.9413调试： (2, 32, 19) (32, 19, 28)
    18/34 [==============>...............] - ETA: 12s - loss: 0.1326 - accuracy: 0.9424调试： (2, 32, 19) (32, 19, 28)
    19/34 [===============>..............] - ETA: 11s - loss: 0.1316 - accuracy: 0.9427调试： (2, 32, 19) (32, 19, 28)
    20/34 [================>.............] - ETA: 10s - loss: 0.1317 - accuracy: 0.9421调试： (2, 4, 19) (4, 19, 28)
    21/34 [=================>............] - ETA: 10s - loss: 0.1320 - accuracy: 0.9422调试： (2, 32, 19) (32, 19, 28)
    22/34 [==================>...........] - ETA: 9s - loss: 0.1318 - accuracy: 0.9421 调试： (2, 32, 19) (32, 19, 28)
    23/34 [===================>..........] - ETA: 8s - loss: 0.1302 - accuracy: 0.9433调试： (2, 32, 19) (32, 19, 28)
    24/34 [====================>.........] - ETA: 7s - loss: 0.1295 - accuracy: 0.9427调试： (2, 32, 19) (32, 19, 28)
    25/34 [=====================>........] - ETA: 6s - loss: 0.1307 - accuracy: 0.9428调试： (2, 32, 19) (32, 19, 28)
    26/34 [=====================>........] - ETA: 6s - loss: 0.1280 - accuracy: 0.9444调试： (2, 32, 19) (32, 19, 28)
    27/34 [======================>.......] - ETA: 5s - loss: 0.1263 - accuracy: 0.9454调试： (2, 32, 19) (32, 19, 28)
    28/34 [=======================>......] - ETA: 4s - loss: 0.1249 - accuracy: 0.9462调试： (2, 32, 19) (32, 19, 28)
    29/34 [========================>.....] - ETA: 3s - loss: 0.1226 - accuracy: 0.9475调试： (2, 32, 19) (32, 19, 28)
    30/34 [=========================>....] - ETA: 3s - loss: 0.1235 - accuracy: 0.9475调试： (2, 32, 19) (32, 19, 28)
    31/34 [==========================>...] - ETA: 2s - loss: 0.1227 - accuracy: 0.9477调试： (2, 32, 19) (32, 19, 28)
    32/34 [===========================>..] - ETA: 1s - loss: 0.1215 - accuracy: 0.9483调试： (2, 32, 19) (32, 19, 28)
    33/34 [============================>.] - ETA: 0s - loss: 0.1213 - accuracy: 0.9477调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    
    Epoch 00002: val_loss improved from 0.13893 to 0.05201, saving model to entity_weights.h5
    34/34 [==============================] - 36s 1s/step - loss: 0.1214 - accuracy: 0.9475 - val_loss: 0.0520 - val_accuracy: 0.9846
    Epoch 3/10
    调试： (2, 32, 19) (32, 19, 28)
     1/34 [..............................] - ETA: 24s - loss: 0.1003 - accuracy: 0.9642调试： (2, 32, 19) (32, 19, 28)
     2/34 [>.............................] - ETA: 23s - loss: 0.0951 - accuracy: 0.9635调试： (2, 32, 19) (32, 19, 28)
     3/34 [=>............................] - ETA: 22s - loss: 0.0779 - accuracy: 0.9719调试： (2, 32, 19) (32, 19, 28)
     4/34 [==>...........................] - ETA: 21s - loss: 0.0767 - accuracy: 0.9735调试： (2, 32, 19) (32, 19, 28)
     5/34 [===>..........................] - ETA: 21s - loss: 0.0761 - accuracy: 0.9745调试： (2, 32, 19) (32, 19, 28)
     6/34 [====>.........................] - ETA: 20s - loss: 0.0845 - accuracy: 0.9685调试： (2, 32, 19) (32, 19, 28)
     7/34 [=====>........................] - ETA: 20s - loss: 0.0869 - accuracy: 0.9674调试： (2, 32, 19) (32, 19, 28)
     8/34 [======>.......................] - ETA: 19s - loss: 0.0854 - accuracy: 0.9656调试： (2, 32, 19) (32, 19, 28)
     9/34 [======>.......................] - ETA: 18s - loss: 0.0799 - accuracy: 0.9684调试： (2, 32, 19) (32, 19, 28)
    10/34 [=======>......................] - ETA: 17s - loss: 0.0756 - accuracy: 0.9696调试： (2, 32, 19) (32, 19, 28)
    11/34 [========>.....................] - ETA: 17s - loss: 0.0737 - accuracy: 0.9703调试： (2, 32, 19) (32, 19, 28)
    12/34 [=========>....................] - ETA: 16s - loss: 0.0705 - accuracy: 0.9717调试： (2, 32, 19) (32, 19, 28)
    13/34 [==========>...................] - ETA: 15s - loss: 0.0689 - accuracy: 0.9730调试： (2, 32, 19) (32, 19, 28)
    14/34 [===========>..................] - ETA: 14s - loss: 0.0683 - accuracy: 0.9740调试： (2, 32, 19) (32, 19, 28)
    15/34 [============>.................] - ETA: 14s - loss: 0.0684 - accuracy: 0.9741调试： (2, 32, 19) (32, 19, 28)
    16/34 [=============>................] - ETA: 13s - loss: 0.0678 - accuracy: 0.9752调试： (2, 32, 19) (32, 19, 28)
    17/34 [==============>...............] - ETA: 12s - loss: 0.0683 - accuracy: 0.9741调试： (2, 32, 19) (32, 19, 28)
    18/34 [==============>...............] - ETA: 11s - loss: 0.0693 - accuracy: 0.9736调试： (2, 32, 19) (32, 19, 28)
    19/34 [===============>..............] - ETA: 11s - loss: 0.0715 - accuracy: 0.9719调试： (2, 32, 19) (32, 19, 28)
    20/34 [================>.............] - ETA: 10s - loss: 0.0714 - accuracy: 0.9723调试： (2, 4, 19) (4, 19, 28)
    21/34 [=================>............] - ETA: 9s - loss: 0.0707 - accuracy: 0.9727 调试： (2, 32, 19) (32, 19, 28)
    22/34 [==================>...........] - ETA: 9s - loss: 0.0732 - accuracy: 0.9717调试： (2, 32, 19) (32, 19, 28)
    23/34 [===================>..........] - ETA: 8s - loss: 0.0730 - accuracy: 0.9715调试： (2, 32, 19) (32, 19, 28)
    24/34 [====================>.........] - ETA: 7s - loss: 0.0729 - accuracy: 0.9713调试： (2, 32, 19) (32, 19, 28)
    25/34 [=====================>........] - ETA: 6s - loss: 0.0734 - accuracy: 0.9705调试： (2, 32, 19) (32, 19, 28)
    26/34 [=====================>........] - ETA: 6s - loss: 0.0727 - accuracy: 0.9709调试： (2, 32, 19) (32, 19, 28)
    27/34 [======================>.......] - ETA: 5s - loss: 0.0718 - accuracy: 0.9713调试： (2, 32, 19) (32, 19, 28)
    28/34 [=======================>......] - ETA: 4s - loss: 0.0711 - accuracy: 0.9716调试： (2, 32, 19) (32, 19, 28)
    29/34 [========================>.....] - ETA: 3s - loss: 0.0707 - accuracy: 0.9717调试： (2, 32, 19) (32, 19, 28)
    30/34 [=========================>....] - ETA: 3s - loss: 0.0705 - accuracy: 0.9718调试： (2, 32, 19) (32, 19, 28)
    31/34 [==========================>...] - ETA: 2s - loss: 0.0693 - accuracy: 0.9723调试： (2, 32, 19) (32, 19, 28)
    32/34 [===========================>..] - ETA: 1s - loss: 0.0693 - accuracy: 0.9718调试： (2, 32, 19) (32, 19, 28)
    33/34 [============================>.] - ETA: 0s - loss: 0.0697 - accuracy: 0.9715调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    
    Epoch 00003: val_loss improved from 0.05201 to 0.04645, saving model to entity_weights.h5
    34/34 [==============================] - 35s 1s/step - loss: 0.0692 - accuracy: 0.9716 - val_loss: 0.0464 - val_accuracy: 0.9837
    Epoch 4/10
    调试： (2, 32, 19) (32, 19, 28)
     1/34 [..............................] - ETA: 24s - loss: 0.0412 - accuracy: 0.9848调试： (2, 32, 19) (32, 19, 28)
     2/34 [>.............................] - ETA: 24s - loss: 0.0351 - accuracy: 0.9891调试： (2, 32, 19) (32, 19, 28)
     3/34 [=>............................] - ETA: 23s - loss: 0.0357 - accuracy: 0.9895调试： (2, 32, 19) (32, 19, 28)
     4/34 [==>...........................] - ETA: 23s - loss: 0.0379 - accuracy: 0.9889调试： (2, 32, 19) (32, 19, 28)
     5/34 [===>..........................] - ETA: 22s - loss: 0.0385 - accuracy: 0.9897调试： (2, 32, 19) (32, 19, 28)
     6/34 [====>.........................] - ETA: 21s - loss: 0.0414 - accuracy: 0.9870调试： (2, 32, 19) (32, 19, 28)
     7/34 [=====>........................] - ETA: 20s - loss: 0.0442 - accuracy: 0.9848调试： (2, 32, 19) (32, 19, 28)
     8/34 [======>.......................] - ETA: 19s - loss: 0.0453 - accuracy: 0.9848调试： (2, 32, 19) (32, 19, 28)
     9/34 [======>.......................] - ETA: 19s - loss: 0.0459 - accuracy: 0.9829调试： (2, 32, 19) (32, 19, 28)
    10/34 [=======>......................] - ETA: 18s - loss: 0.0461 - accuracy: 0.9819调试： (2, 32, 19) (32, 19, 28)
    11/34 [========>.....................] - ETA: 17s - loss: 0.0467 - accuracy: 0.9815调试： (2, 32, 19) (32, 19, 28)
    12/34 [=========>....................] - ETA: 16s - loss: 0.0499 - accuracy: 0.9808调试： (2, 32, 19) (32, 19, 28)
    13/34 [==========>...................] - ETA: 16s - loss: 0.0502 - accuracy: 0.9801调试： (2, 32, 19) (32, 19, 28)
    14/34 [===========>..................] - ETA: 15s - loss: 0.0496 - accuracy: 0.9806调试： (2, 32, 19) (32, 19, 28)
    15/34 [============>.................] - ETA: 14s - loss: 0.0487 - accuracy: 0.9814调试： (2, 32, 19) (32, 19, 28)
    16/34 [=============>................] - ETA: 13s - loss: 0.0482 - accuracy: 0.9818调试： (2, 32, 19) (32, 19, 28)
    17/34 [==============>...............] - ETA: 13s - loss: 0.0475 - accuracy: 0.9822调试： (2, 32, 19) (32, 19, 28)
    18/34 [==============>...............] - ETA: 12s - loss: 0.0462 - accuracy: 0.9832调试： (2, 32, 19) (32, 19, 28)
    19/34 [===============>..............] - ETA: 11s - loss: 0.0453 - accuracy: 0.9836调试： (2, 32, 19) (32, 19, 28)
    20/34 [================>.............] - ETA: 10s - loss: 0.0462 - accuracy: 0.9829调试： (2, 4, 19) (4, 19, 28)
    21/34 [=================>............] - ETA: 9s - loss: 0.0456 - accuracy: 0.9833 调试： (2, 32, 19) (32, 19, 28)
    22/34 [==================>...........] - ETA: 9s - loss: 0.0447 - accuracy: 0.9838调试： (2, 32, 19) (32, 19, 28)
    23/34 [===================>..........] - ETA: 8s - loss: 0.0452 - accuracy: 0.9832调试： (2, 32, 19) (32, 19, 28)
    24/34 [====================>.........] - ETA: 7s - loss: 0.0449 - accuracy: 0.9833调试： (2, 32, 19) (32, 19, 28)
    25/34 [=====================>........] - ETA: 6s - loss: 0.0447 - accuracy: 0.9833调试： (2, 32, 19) (32, 19, 28)
    26/34 [=====================>........] - ETA: 6s - loss: 0.0466 - accuracy: 0.9820调试： (2, 32, 19) (32, 19, 28)
    27/34 [======================>.......] - ETA: 5s - loss: 0.0468 - accuracy: 0.9819调试： (2, 32, 19) (32, 19, 28)
    28/34 [=======================>......] - ETA: 4s - loss: 0.0463 - accuracy: 0.9822调试： (2, 32, 19) (32, 19, 28)
    29/34 [========================>.....] - ETA: 3s - loss: 0.0454 - accuracy: 0.9829调试： (2, 32, 19) (32, 19, 28)
    30/34 [=========================>....] - ETA: 3s - loss: 0.0456 - accuracy: 0.9825调试： (2, 32, 19) (32, 19, 28)
    31/34 [==========================>...] - ETA: 2s - loss: 0.0454 - accuracy: 0.9823调试： (2, 32, 19) (32, 19, 28)
    32/34 [===========================>..] - ETA: 1s - loss: 0.0450 - accuracy: 0.9825调试： (2, 32, 19) (32, 19, 28)
    33/34 [============================>.] - ETA: 0s - loss: 0.0444 - accuracy: 0.9828调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 9, 19) (9, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    调试： (2, 32, 19) (32, 19, 28)
    
    Epoch 00004: val_loss improved from 0.04645 to 0.04206, saving model to entity_weights.h5
    34/34 [==============================] - 35s 1s/step - loss: 0.0444 - accuracy: 0.9827 - val_loss: 0.0421 - val_accuracy: 0.9858
    Epoch 5/10
    调试： (2, 32, 19) (32, 19, 28)
     1/34 [..............................] - ETA: 25s - loss: 0.0393 - accuracy: 0.9857调试： (2, 32, 19) (32, 19, 28)
     2/34 [>.............................] - ETA: 24s - loss: 0.0553 - accuracy: 0.9816调试： (2, 32, 19) (32, 19, 28)
     3/34 [=>............................] - ETA: 23s - loss: 0.0487 - accuracy: 0.9846调试： (2, 32, 19) (32, 19, 28)
     4/34 [==>...........................] - ETA: 23s - loss: 0.0454 - accuracy: 0.9843调试： (2, 32, 19) (32, 19, 28)
     5/34 [===>..........................] - ETA: 22s - loss: 0.0418 - accuracy: 0.9852调试： (2, 32, 19) (32, 19, 28)
     6/34 [====>.........................] - ETA: 21s - loss: 0.0408 - accuracy: 0.9854调试： (2, 32, 19) (32, 19, 28)
     7/34 [=====>........................] - ETA: 20s - loss: 0.0393 - accuracy: 0.9867调试： (2, 32, 19) (32, 19, 28)
     8/34 [======>.......................] - ETA: 20s - loss: 0.0373 - accuracy: 0.9877调试： (2, 32, 19) (32, 19, 28)
     9/34 [======>.......................] - ETA: 19s - loss: 0.0381 - accuracy: 0.9873调试： (2, 32, 19) (32, 19, 28)
    10/34 [=======>......................] - ETA: 18s - loss: 0.0372 - accuracy: 0.9883调试： (2, 32, 19) (32, 19, 28)
    11/34 [========>.....................] - ETA: 17s - loss: 0.0364 - accuracy: 0.9877调试： (2, 32, 19) (32, 19, 28)
    12/34 [=========>....................] - ETA: 16s - loss: 0.0371 - accuracy: 0.9878调试： (2, 32, 19) (32, 19, 28)
    13/34 [==========>...................] - ETA: 16s - loss: 0.0366 - accuracy: 0.9881调试： (2, 32, 19) (32, 19, 28)
    14/34 [===========>..................] - ETA: 15s - loss: 0.0351 - accuracy: 0.9887调试： (2, 32, 19) (32, 19, 28)
    15/34 [============>.................] - ETA: 14s - loss: 0.0346 - accuracy: 0.9888调试： (2, 32, 19) (32, 19, 28)
    16/34 [=============>................] - ETA: 13s - loss: 0.0357 - accuracy: 0.9877调试： (2, 32, 19) (32, 19, 28)
    17/34 [==============>...............] - ETA: 12s - loss: 0.0345 - accuracy: 0.9883调试： (2, 32, 19) (32, 19, 28)
    

## 9、Core解析（数据、训练）

### 9.1、Core相关数据解析


```python
from rasa.core.train import get_no_of_stories
from rasa.core.domain import TemplateDomain
from rasa.core.training.dsl import StoryFileReader
from rasa.core.training.structures import Story,StoryGraph,StoryStep,StoryStringHelper

story_file = 'data/stories.md'
stories = await StoryFileReader.read_from_folder(story_file, 
                               TemplateDomain.load(domain_file)) #返回Story类的元素List[StoryStep]

stories_generate = Story(story_steps=stories, story_name='my_stories') #Story类中包含StoryStep--即单个story
# <rasa.core.training.structures.Story at 0x2eb464dee08>
print(stories_generate.__dict__)

storyStep_0 = stories[0]
print(storyStep_0.__dict__)
print(storyStep_0.events[0].__dict__)
print(storyStep_0.events[4].__dict__)

storyStep_2 = stories[2]
print(storyStep_2.__dict__)
print(storyStep_2.events[0].__dict__)
storyStep_2.events[14].__dict__

no_stories = await get_no_of_stories('data/stories.md', domain_file)
#no_stories = len(stories)
storyGraph = StoryGraph(stories)
print(storyGraph.__dict__)
```

### 9.2、stories训练数据扩增方法解析


```python
##加载stories文件，生成训练数据
from rasa.importers.importer import TrainingDataImporter
from rasa.core.training import load_data
from rasa.core.training.generator import (
    TrainingDataGenerator,                                 
    TrackerLookupDict,TrackersTuple,
    TrackerWithCachedStates
)

trainingDataImporter = TrainingDataImporter().load_from_config(configs_file,
                                           domain_file,training_files)

await trainingDataImporter.get_config()
await trainingDataImporter.get_nlu_data()

trainDataGenerator = TrainingDataGenerator(storyGraph, domain)
trainDataGenerator.__dict__
'''
{'story_graph': <rasa.core.training.structures.c at 0x2eb47d88348>,
 'domain': <rasa.core.domain.Domain at 0x2eb46595d08>,
 'config': ExtractorConfig(remove_duplicates=True, unique_last_num_states=None, augmentation_factor=50, max_number_of_augmented_trackers=500, tracker_limit=None, use_story_concatenation=True, rand=<random.Random object at 0x000002EAFE7BB0F8>),
 'hashed_featurizations': set()}
'''
list_TrackerWithCachedStates = trainDataGenerator.generate()
len(list_TrackerWithCachedStates) #522
trackerWithCachedStates = list_TrackerWithCachedStates[100]
trackerWithCachedStates.__dict__
agent.train(list_TrackerWithCachedStates)

##story数据扩增解析
#from  rasa.core.training.generator import generate
from rasa.core.training.dsl import StoryFileReader
from rasa.core.training.structures import Story,StoryGraph,StoryStep
from rasa.core.training.generator import TrainingDataGenerator
from rasa.core.domain import TemplateDomain

story_file = 'data_1/stories_4.md'
stories = await StoryFileReader.read_from_folder(story_file, 
                               TemplateDomain.load(domain_file)) #返回Story类的元素List[StoryStep]
storyGraph = StoryGraph(stories)
storyGraph.__dict__ #保存到storyGraph.txt中

trainDataGenerator = TrainingDataGenerator(storyGraph, domain)
list_TrackerWithCachedStates = trainDataGenerator.generate()
len(list_TrackerWithCachedStates) #244
trackerWithCachedStates = list_TrackerWithCachedStates[4]
trackerWithCachedStates.__dict__
trackerWithCachedStates.export_stories()

export_stories_path = r'parsed_datas\stories augment\augmented_stories.md'
for tr in list_TrackerWithCachedStates:
    tr.export_stories_to_file(export_stories_path)
```

### 9.3、Policy训练解析

policy_ensemble生成


```python
ensemble = agent._create_ensemble(policies)
ensemble.__dict__
ensemble.policies
'''
[<rasa.core.policies.two_stage_fallback.TwoStageFallbackPolicy at 0x20b41124a48>,
 <rasa.core.policies.memoization.MemoizationPolicy at 0x20b410d3fc8>,
 <rasa.core.policies.form_policy.FormPolicy at 0x20b411249c8>,
 <rasa.core.policies.mapping_policy.MappingPolicy at 0x20b41124108>,
 <rasa.core.policies.keras_policy.KerasPolicy at 0x20b41207a88>]
'''
twoStageFallbackPolicy = ensemble.policies[0]
twoStageFallbackPolicy.__dict__
memoizationPolicy = ensemble.policies[1]
memoizationPolicy.__dict__
formPolicy = ensemble.policies[2]
formPolicy.__dict__
mappingPolicy = ensemble.policies[3]
mappingPolicy.__dict__
kerasPolicy = ensemble.policies[4]
kerasPolicy.__dict__
kerasPolicy.train(list_TrackerWithCachedStates, domain)
```

keras_policy 单独训练


```python
###keras_policy train
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer

maxHistoryTrackerFeaturizer = kerasPolicy.featurizer
maxHistoryTrackerFeaturizer.use_intent_probabilities = True

training_data = kerasPolicy.featurize_for_training(list_TrackerWithCachedStates, domain)
training_data.y.shape #(1266, 74)
training_data.X.shape #(1266, 5, 135)
training_data.max_history()  #5

list_TrackerWithCachedStates[0].events
list_TrackerWithCachedStates[0].as_dialogue().as_dict()
states = maxHistoryTrackerFeaturizer._create_states(list_TrackerWithCachedStates[0], domain)
trackers_as_states = maxHistoryTrackerFeaturizer.prediction_states(list_TrackerWithCachedStates, domain)
tracker_0 = list_TrackerWithCachedStates[0]
tracker_0.export_stories()
tracker_0.export_stories_to_file('tracker_0.json')
```

ted_policy单独训练


```python
tEDPolicy = ensemble.policies[5]
tEDPolicy.__dict__
tEDPolicy.train(list_TrackerWithCachedStates, domain)
tEDPolicy
label_data = tEDPolicy._create_label_data(domain)
label_data.__dict__
```
