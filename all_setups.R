batch_size = c(4, 16)
seq_length = c(50, 200)
learning_rate = c(1e-3, 1e-5)
embedding_dim = c(256, 512)
lstm_units = c(256, 1024)

cont = 0
x = matrix(nrow=64/2, ncol=6)
colnames(x) = c('idx','batch_size','seq_length','learning_rate','embedding_dim','lstm_units')

for (v in 1:length(lstm_units)){
  for (h in 1:length(embedding_dim)){
    for (k in 1:length(learning_rate)){
      for (j in 1:length(seq_length)){
        for (i in 1:length(batch_size)){
          cont = cont + 1
          x[cont,] = c(cont, 
                       batch_size[i], 
                       seq_length[j], 
                       learning_rate[k], 
                       embedding_dim[h], 
                       lstm_units[v])
        }
      }
    }
  }
}
