import torch

path = 'outputs/data/hytrel/pretraining_10M_tables/hytrel-llama/v2-10M_only_query-not_freeze_gnn-trained_roberta_base-freeze_llama-10.cfg/200K/model.bin'

state_dict = torch.load(path)

state_dict = {k[8:]: v for k,v in state_dict.items() if k.startswith('gformer')}

torch.save(state_dict, path.replace('model.bin', 'sqformer_model.bin'))