# TCC_NMH
Este repositório é referente ao meu Trabalho de Conclusão de Curso do curso de graduação de Bacharelado em Estatística na UFRGS. O material está disponível no LUME UFRGS com o título de ![Composição Automática de Músicas utilizando Redes Neurais Recorrentes](https://lume.ufrgs.br/handle/10183/261989).


## Inspiração

O material desenvolvido foi inspirado em um laboratório de geração de músicas utilizando Tensorflow disponibilizado pelo MIT e por um artigo sobre composição musical utilizando Redes Neurais Recorrentes da Universidade de Stanford:

* https://goodboychan.github.io/python/tensorflow/mit/2021/02/14/music-generation.html
* https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/2762076.pdf


## Criando Ambiente CONDA

```
conda create --name tcc python==3.9.12
conda activate tcc
pip install -r requirements.txt
```


## Scripts

* `script.ipynb`: notebook responsável pela leitura, pré-processamento, ajuste do modelo, composição musical e armazenamento dos resultados em formato *.json*.
* `abcnotation_crawler.py`: programa responsável por fazer a coleta de dados, via web crawling e web scraping, das músicas em formato *.abc* do site ![abcnotation.com](https://abcnotation.com).
* `abc_to_midi.ipynb`: notebook responsável por converter os arquivos *.abc* para MIDI e, caso a conversão ocorra, temos uma música gerada pelo modelo.
* `analysis.R`: construção dos gráficos com base nos resultados obtidos.
* `all_setups.R`: mapeamento de todas as configurações de hiperparâmetros utilizadas nos modelos.


## Resultados

As músicas resultantes do trabalho, geradas pelos modelos treinados em ambas as bases de dados, estão disponíveis no Youtube:

* Irish: https://www.youtube.com/watch?v=oL_zslqNkrk
* ABC Notation: https://www.youtube.com/watch?v=wPzcATIRNTA