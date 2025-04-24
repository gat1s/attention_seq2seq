# attention_seq2seq

We set up a toy French to English translation task, __implement the attentional seq2seq architecture of [Leung et al. (2015)](https://arxiv.org/pdf/1508.04025)__ in PyTorch, wrap it into a training loop and train a model, after some experimentation:

<img src="results/5e-03_0.05_32_{'h_size':90,'dropout':0.2,'n_layers':2,'att_method':'general','c':'final_model'}.png" alt="pic" width="300"/>
                  
We can then feed French into the model and use beam search to get its take on some probable translations:

<img src="write-up_pics/c.png" alt="pic" width="300"/>

