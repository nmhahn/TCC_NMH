
# libraries ---------------------------------------------------------------
library(here)
library(rjson)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggcorrplot)
library(patchwork)
library(knitr)
library(kableExtra)


# load data from json -----------------------------------------------------
 
 # pt1 - all tanh models
dt = data.frame(
  data_source = NA,
  idx = NA,
  epochs = NA,
  batch_size = NA,
  seq_length = NA,
  learning_rate = NA,
  vocab_size = NA,
  embedding_dim = NA,
  lstm_units = NA,
  train_cross_entropy_loss = NA,
  train_perplexity_exp = NA,
  test_cross_entropy_loss = NA,
  test_perplexity_exp = NA,
  generated_songs_abc = NA
  )
dt$train_loss_history = list(list())
dt_aux = dt

for(name in c('irish','abcnotation')){
  for(i in 1:32){
    data = fromJSON(
      file=here('scripts','models','tanh', 
                sprintf('model_%s_%02d', name, i),
                sprintf('results_%s_%02d.json', name, i)
                )
    )
    
    dt_aux['data_source'] = name
    dt_aux['idx'] = i
    dt_aux['epochs'] = data$epochs
    dt_aux['batch_size'] = data$batch_size
    dt_aux['seq_length'] = data$seq_length
    dt_aux['learning_rate'] = data$learning_rate
    dt_aux['vocab_size'] = data$vocab_size
    dt_aux['embedding_dim'] = data$embedding_dim
    dt_aux['lstm_units'] = data$lstm_units
    dt_aux['train_cross_entropy_loss'] = data$`train categorical cross-entropy loss`
    dt_aux['train_perplexity_exp'] = data$`train perplexity (exp)`
    dt_aux['test_cross_entropy_loss'] = data$`test categorical cross-entropy loss`
    dt_aux['test_perplexity_exp'] = data$`test perplexity (exp)`
    dt_aux['generated_songs_abc'] = paste(data$generated_songs_abc, collapse = '\n\n')
    dt_aux['train_loss_history'] = list(list(data$`train loss history (from plot)`))
    dt = rbind(dt, dt_aux) 
  }
}
dt = dt[2:nrow(dt),]
dt$pos = 1:nrow(dt)
rownames(dt) = dt$pos
dt_tanh = dt

  # pt2 - sigmoid models (top 2 and bottom 2 perplexity)
dt = data.frame(
  data_source = NA,
  idx = NA,
  epochs = NA,
  batch_size = NA,
  seq_length = NA,
  learning_rate = NA,
  vocab_size = NA,
  embedding_dim = NA,
  lstm_units = NA,
  train_cross_entropy_loss = NA,
  train_perplexity_exp = NA,
  test_cross_entropy_loss = NA,
  test_perplexity_exp = NA,
  generated_songs_abc = NA
)
dt$train_loss_history = list(list())
dt_aux = dt
rownames(dt) = dt$pos

  # irish
for(name in c('irish','abcnotation')){
  vetor = if(name=='irish'){c(5,7,11,25)} else {c(6,8,20,28)}
  for(i in vetor){
    data = fromJSON(
      file=here('scripts','models','sigmoid', 
                sprintf('model_%s_%02d', name, i),
                sprintf('results_%s_%02d.json', name, i)
      )
    )
    
    dt_aux['data_source'] = name
    dt_aux['idx'] = i
    dt_aux['epochs'] = data$epochs
    dt_aux['batch_size'] = data$batch_size
    dt_aux['seq_length'] = data$seq_length
    dt_aux['learning_rate'] = data$learning_rate
    dt_aux['vocab_size'] = data$vocab_size
    dt_aux['embedding_dim'] = data$embedding_dim
    dt_aux['lstm_units'] = data$lstm_units
    dt_aux['train_cross_entropy_loss'] = data$`train categorical cross-entropy loss`
    dt_aux['train_perplexity_exp'] = data$`train perplexity (exp)`
    dt_aux['test_cross_entropy_loss'] = data$`test categorical cross-entropy loss`
    dt_aux['test_perplexity_exp'] = data$`test perplexity (exp)`
    dt_aux['generated_songs_abc'] = paste(data$generated_songs_abc, collapse = '\n\n')
    dt_aux['train_loss_history'] = list(list(data$`train loss history (from plot)`))
    dt = rbind(dt, dt_aux) 
  }
}
dt = dt[2:nrow(dt),]
dt$pos = 1:nrow(dt)
dt_sigmoid = dt


dt_loss = dt_tanh[,c(1:9,15)] %>% unnest(train_loss_history) %>%
  mutate(
    epochs=rep(1:2000,64),
    idx=as.factor(idx),
    act='tanh'
  )


dt_loss2 = dt_sigmoid[,c(1:9,15)] %>% unnest(train_loss_history) %>%
  mutate(
    epochs=rep(1:2000,8),
    idx=as.factor(idx),
    act='sigmoid'
  )

# plots -------------------------------------------------------------------

plot_rslt = function(df, group, title, colors, legend='none', clear_axis_x=F, clear_axis_y=F){
  g = ggplot(data=df, aes(x=epochs, y=train_loss_history, colour=.data[[group]])) +
    geom_line() +
    scale_colour_manual(values=colors) +
    xlab('Época') + ylab('Perda') +
    xlim(c(0,2000)) + ylim(c(0,6)) +
    annotate(geom='label', label='idx =      |      ', 
             x=1000, y=5.5, hjust=0, vjust=0.30, size=4.5) +
    annotate(geom='text', label=title[1], x=1000, y=5.5, hjust=-2.5, vjust=0.15, size=4.5, color=colors[1]) +
    annotate(geom='text', label=title[2], x=1000, y=5.5, hjust=-4.2, vjust=0.15, size=4.5, color=colors[2]) +
    theme_bw() +
    theme(
      axis.title = element_text(size=10),
      plot.margin=unit(c(0.05,0.1,0.05,0.05), 'lines'),
      legend.position=legend
    )
  if(clear_axis_x==T){
    g = g + theme(
      axis.text.x = element_blank(), 
      axis.ticks.x = element_blank(),
      axis.title.x = element_blank()
    )
  }
  if(clear_axis_y==T){
    g = g + theme(
      axis.text.y = element_blank(), 
      axis.ticks.y = element_blank(),
      axis.title.y = element_blank()
    )
  }
  return(g)
}

cores = c('darkgreen','#6F02B5')
grupo = 'idx'

# pt1 - tanh

    # irish
fonte = 'irish'


g1 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(1,5)), group=grupo, 
               title=c('01','05'), color=cores, clear_axis_x=T)
g2 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(2,6)), group=grupo, 
               title=c('02','06'), color=cores, clear_axis_x=T, clear_axis_y=T)
g3 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(3,7)), group=grupo, 
               title=c('03','07'), color=cores, clear_axis_x=T, clear_axis_y=T)
g4 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(4,8)), group=grupo, 
               title=c('04','08'), color=cores, clear_axis_x=T, clear_axis_y=T)
g5 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(9,13)), group=grupo, 
               title=c('09','13'), color=cores, clear_axis_x=T)
g6 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(10,14)), group=grupo, 
               title=c('10','14'), color=cores, clear_axis_x=T, clear_axis_y=T)
g7 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(11,15)), group=grupo, 
               title=c('11','15'), color=cores, clear_axis_x=T, clear_axis_y=T)
g8 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(12,16)), group=grupo, 
               title=c('12','16'), color=cores, clear_axis_x=T, clear_axis_y=T)
g9 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(17,21)), group=grupo, 
               title=c('17','21'), color=cores, clear_axis_x=T)
g10 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(18,22)), group=grupo, 
                title=c('18','22'), color=cores, clear_axis_x=T, clear_axis_y=T)
g11 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(19,23)), group=grupo, 
                title=c('19','23'), color=cores, clear_axis_x=T, clear_axis_y=T)
g12 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(20,24)), group=grupo, 
                title=c('20','24'), color=cores, clear_axis_x=T, clear_axis_y=T)
g13 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(25,29)), group=grupo, 
                title=c('25','29'), color=cores)
g14 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(26,30)), group=grupo, 
                title=c('26','30'), color=cores, clear_axis_y=T)
g15 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(27,31)), group=grupo, 
                title=c('27','31'), color=cores, clear_axis_y=T)
g16 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(28,32)), group=grupo, 
                title=c('28','32'), color=cores, clear_axis_y=T)

g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16 +
  plot_layout(ncol=4)




    # abcnotation
fonte = 'abcnotation'


g1 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(1,5)), group=grupo, 
               title=c('01','05'), color=cores, clear_axis_x=T)
g2 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(2,6)), group=grupo, 
               title=c('02','06'), color=cores, clear_axis_x=T, clear_axis_y=T)
g3 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(3,7)), group=grupo, 
               title=c('03','07'), color=cores, clear_axis_x=T, clear_axis_y=T)
g4 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(4,8)), group=grupo, 
               title=c('04','08'), color=cores, clear_axis_x=T, clear_axis_y=T)
g5 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(9,13)), group=grupo, 
               title=c('09','13'), color=cores, clear_axis_x=T)
g6 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(10,14)), group=grupo, 
               title=c('10','14'), color=cores, clear_axis_x=T, clear_axis_y=T)
g7 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(11,15)), group=grupo, 
               title=c('11','15'), color=cores, clear_axis_x=T, clear_axis_y=T)
g8 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(12,16)), group=grupo, 
               title=c('12','16'), color=cores, clear_axis_x=T, clear_axis_y=T)
g9 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(17,21)), group=grupo, 
               title=c('17','21'), color=cores, clear_axis_x=T)
g10 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(18,22)), group=grupo, 
                title=c('18','22'), color=cores, clear_axis_x=T, clear_axis_y=T)
g11 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(19,23)), group=grupo, 
                title=c('19','23'), color=cores, clear_axis_x=T, clear_axis_y=T)
g12 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(20,24)), group=grupo, 
                title=c('20','24'), color=cores, clear_axis_x=T, clear_axis_y=T)
g13 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(25,29)), group=grupo, 
                title=c('25','29'), color=cores)
g14 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(26,30)), group=grupo, 
                title=c('26','30'), color=cores, clear_axis_y=T)
g15 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(27,31)), group=grupo, 
                title=c('27','31'), color=cores, clear_axis_y=T)
g16 = plot_rslt(dt_loss %>% filter(data_source==fonte, idx %in% c(28,32)), group=grupo, 
                title=c('28','32'), color=cores, clear_axis_y=T)

g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16 +
  plot_layout(ncol=4)




  # correlation matrix parameters, loss and perplexity (tanh)
    # irish
dt_aux = dt_tanh[,-c(14,15)] %>% filter(data_source=='irish') %>% select(
  lstm_units, embedding_dim, learning_rate, seq_length, batch_size,
  train_cross_entropy_loss, test_perplexity_exp
  ) %>% rename(
    train_loss = train_cross_entropy_loss,
    test_perplexity = test_perplexity_exp
  )
corr = round(cor(dt_aux, method='spearman'), 2)
ggcorrplot(corr[,6:7], lab=T, title = 'Correlação Spearman - Irish', 
           ggtheme = theme_void() + theme(plot.title=element_text(hjust=0.5), plot.margin=unit(c(0,0.5,0,0.5), 'lines'))
           )
    # abcnotation
dt_aux = dt_tanh[,-c(14,15)] %>% filter(data_source=='abcnotation') %>% select(
  lstm_units, embedding_dim, learning_rate, seq_length, batch_size,
  train_cross_entropy_loss, test_perplexity_exp
) %>% rename(
  train_loss = train_cross_entropy_loss,
  test_perplexity = test_perplexity_exp
)
corr = round(cor(dt_aux, method='spearman'), 2)
ggcorrplot(corr[,6:7], lab=T, title = 'Correlação Spearman - ABC Notation', 
           ggtheme = theme_void() + theme(plot.title=element_text(hjust=0.5), plot.margin=unit(c(0,0.5,0,0.5), 'lines'))
)




  # pt2 - sigmoid
plot_rslt2 = function(df, group, title, colors, legend='none', clear_axis_x=F, clear_axis_y=F){
  g = ggplot(data=df, aes(x=epochs, y=train_loss_history, colour=.data[[group]])) +
    geom_line() +
    scale_colour_manual(values=colors) +
    xlab('Época') + ylab('Perda') +
    xlim(c(0,2000)) + ylim(c(0,6)) +
    annotate(geom='label', label=title, 
             x=1500, y=5.5, hjust=0, vjust=0.30, size=4.5) +
    theme_bw() +
    theme(
      axis.title = element_text(size=10),
      plot.margin=unit(c(0.05,0.1,0.05,0.05), 'lines'),
      legend.position=legend
    )
  if(clear_axis_x==T){
    g = g + theme(
      axis.text.x = element_blank(), 
      axis.ticks.x = element_blank(),
      axis.title.x = element_blank()
    )
  }
  if(clear_axis_y==T){
    g = g + theme(
      axis.text.y = element_blank(), 
      axis.ticks.y = element_blank(),
      axis.title.y = element_blank()
    )
  }
  return(g)
}

cores = c('#6F02B5','darkgreen')
grupo = 'act'

    # irish
fonte = 'irish'
g1 = plot_rslt2(
  df=rbind(
    dt_loss %>% filter(data_source==fonte, idx==5),
    dt_loss2 %>% filter(data_source==fonte, idx==5)
  ), 
  group=grupo, title='idx = 05', color=cores, clear_axis_x=T)
g2 = plot_rslt2(
  df=rbind(
    dt_loss %>% filter(data_source==fonte, idx==7),
    dt_loss2 %>% filter(data_source==fonte, idx==7)
  ), 
  group=grupo, title='idx = 07', color=cores, clear_axis_x=T, clear_axis_y=T)
g3 = plot_rslt2(
  df=rbind(
    dt_loss %>% filter(data_source==fonte, idx==11),
    dt_loss2 %>% filter(data_source==fonte, idx==11)
  ), 
  group=grupo, title='idx = 11', color=cores)
g4 = plot_rslt2(
  df=rbind(
    dt_loss %>% filter(data_source==fonte, idx==25),
    dt_loss2 %>% filter(data_source==fonte, idx==25)
  ), 
  group=grupo, title='idx = 25', color=cores, clear_axis_y=T)

g1+g2+g3+g4 +
  plot_layout(ncol=2)
  

    # abcnotation
fonte = 'abcnotation'
g1 = plot_rslt2(
  df=rbind(
    dt_loss %>% filter(data_source==fonte, idx==8),
    dt_loss2 %>% filter(data_source==fonte, idx==8)
  ), 
  group=grupo, title='idx = 08', color=cores, clear_axis_x=T)
g2 = plot_rslt2(
  df=rbind(
    dt_loss %>% filter(data_source==fonte, idx==6),
    dt_loss2 %>% filter(data_source==fonte, idx==6)
  ), 
  group=grupo, title='idx = 06', color=cores, clear_axis_x=T, clear_axis_y=T)
g3 = plot_rslt2(
  df=rbind(
    dt_loss %>% filter(data_source==fonte, idx==20),
    dt_loss2 %>% filter(data_source==fonte, idx==20)
  ), 
  group=grupo, title='idx = 20', color=cores)
g4 = plot_rslt2(
  df=rbind(
    dt_loss %>% filter(data_source==fonte, idx==28),
    dt_loss2 %>% filter(data_source==fonte, idx==28)
  ), 
  group=grupo, title='idx = 28', color=cores, clear_axis_y=T)


g1+g2+g3+g4 +
  plot_layout(ncol=2)


# tables ------------------------------------------------------------------

plot_table = function(df){
  return(
    kable(
      x=df, 
      format='latex',
      row.names=F,
      align='c',
      digits=3,
      caption='mytable',
      booktabs=T
    ) %>% kable_styling(
      latex_options=c('scale_down')
    )
  )
}

  # pt 1 - tanh
dt = dt_tanh
dt$learning_rate = as.character(dt$learning_rate)
colunas = c('idx','lstm_units','embedding_dim','learning_rate','seq_length','batch_size',
            'train_cross_entropy_loss','test_perplexity_exp')

plot_table(df=dt[1:32, colunas]) # irish
plot_table(df=dt[33:64, colunas]) # abcnotation


  # pt 2 - sigmoid
dt = dt_sigmoid
dt$learning_rate = as.character(dt$learning_rate)

    # irish
ids = c(5,7,11,25)
dt_aux = rbind(
  dt_tanh[,-c(14,15)] %>% filter(data_source=='irish', idx %in% ids) %>%
  select(all_of(colunas)) %>% mutate(act='tanh'),
  dt_sigmoid[,-c(14,15)] %>% filter(data_source=='irish', idx %in% ids) %>%
    select(all_of(colunas)) %>% mutate(act='sigmoid')
  )

dt_aux = data.frame(
  idx = ids,
  tanh_train = round(dt_aux %>% filter(act=='tanh') %>% select(train_cross_entropy_loss),3)[,1],
  sigmoid_train = round(dt_aux %>% filter(act=='sigmoid') %>% select(train_cross_entropy_loss),3)[,1],
  tanh_perp = round(dt_aux %>% filter(act=='tanh') %>% select(test_perplexity_exp),3)[,1],
  sigmoid_perp = round(dt_aux %>% filter(act=='sigmoid') %>% select(test_perplexity_exp),3)[,1]
)
plot_table(df=dt_aux)

# plot_table(df=dt[1:4, colunas]) # irish

    # abcnotation
ids = c(8,6,20,28)
dt_aux = rbind(
  dt_tanh[,-c(14,15)] %>% filter(data_source=='abcnotation', idx %in% ids) %>%
    select(all_of(colunas)) %>% mutate(act='tanh'),
  dt_sigmoid[,-c(14,15)] %>% filter(data_source=='abcnotation', idx %in% ids) %>%
    select(all_of(colunas)) %>% mutate(act='sigmoid')
)

dt_aux = data.frame(
  idx = ids,
  tanh_train = round(dt_aux %>% filter(act=='tanh') %>% select(train_cross_entropy_loss),3)[,1],
  sigmoid_train = round(dt_aux %>% filter(act=='sigmoid') %>% select(train_cross_entropy_loss),3)[,1],
  tanh_perp = round(dt_aux %>% filter(act=='tanh') %>% select(test_perplexity_exp),3)[,1],
  sigmoid_perp = round(dt_aux %>% filter(act=='sigmoid') %>% select(test_perplexity_exp),3)[,1]
)
plot_table(df=dt_aux)



# plot_table(df=dt[5:8, colunas]) # abcnotation


# -------------------------------------------------------------------------

sigmoid = function(x) {1 / (1 + exp(-x))}
dt_aux = data.frame(
  x = seq(from=-5, to=5, length=100)
) %>% mutate(y = sigmoid(x))
ggplot(data=dt_aux, aes(x=x, y=y)) +
  geom_line() +
  xlab('x') + ylab('y') +
  theme_bw()


tanh = function(x) {(exp(x) - exp(-x)) / (exp(x) + exp(-x))}
dt_aux = data.frame(
  x = seq(from=-5, to=5, length=100)
) %>% mutate(y = tanh(x))
ggplot(data=dt_aux, aes(x=x, y=y)) +
  geom_line() +
  xlab('x') + ylab('y') +
  theme_bw()



# -------------------------------------------------------------------------

# # top 2 best and top 2 worst train_perplexity_exp
# dt = dt_tanh
#   # irish
# dt_aux = dt[dt$data_source=='irish',c(1,2,4:9,13)]
# head(dt_aux[order(dt_aux$test_perplexity_exp, decreasing = F),],2)
# tail(dt_aux[order(dt_aux$test_perplexity_exp, decreasing = F),],2)
#   # abcnotation
# dt_aux = dt[dt$data_source=='abcnotation',c(1,2,4:9,13)]
# head(dt_aux[order(dt_aux$test_perplexity_exp, decreasing = F),],2)
# tail(dt_aux[order(dt_aux$test_perplexity_exp, decreasing = F),],2)

