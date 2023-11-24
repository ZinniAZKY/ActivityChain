# ActivityChain
This is the generative model build of PT based activity chain.

In tokenizerGenaration.py, we create 33 unique tokens of activity chain. Inside the EOS, PAD and UNK do not exists in the input dataset.

In Tripchain.py, we transform the original OD trajectories into activity chain of 15 minutes time series. Notice that '@' in the input data need to be replaced by '_' to use this tokenizer, and in Osaka, the actibity number of 46 and 47 in PT original data needed to be changed manually.

ValSeparate.py is used for generating input dataset of training and validation.

CustomGPT1.py, CustomGRU.py and CustomLSTM.py are used for performance comparision of the prediction results as the 3 main model.

Embeddings.py deals with the semantics information of each PT agent.

Pretrain.py contains the main training loop and validation loop of using the main model.

Finetuning.py aims to combine the pretrained model based on PT data with other GPS data like ETC2.0 or Agoop.
