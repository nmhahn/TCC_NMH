%Classe padrão para trabalho de conclusão do Curso de Bacharelado em Estatística.
%Desenvolvido por: Emerson Christ Boeira - adaptado de DeLeTeX.
%Adaptado pela Comissão de Graduação em Estatística

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{automatextcc}[15/09/2017 TCC Curso de Bacharelado em Estatística]

\LoadClass[12pt,a4paper]{report}
\RequirePackage[brazil]{babel}
\RequirePackage{indentfirst}
%\RequirePackage[utf8x]{inputenc}
\RequirePackage{times}
\RequirePackage{natbib}
\RequirePackage{graphicx}
\RequirePackage{amssymb,amsfonts,amsmath,amsthm,euscript}
\RequirePackage{textcomp}
\RequirePackage[T1]{fontenc}%
\RequirePackage[unicode=true, colorlinks = true, linkcolor=blue, citecolor=blue]{hyperref}
\RequirePackage{float} % Para posicionar as figuras de forma mais conveniente, uma alternativa é o pacote placeins
\RequirePackage{mathrsfs}%
\RequirePackage{longtable}%
\RequirePackage{setspace}
\RequirePackage{geometry, calc, color}
\RequirePackage{wrapfig}%
\RequirePackage{subfigure}
\RequirePackage{tabularx}
\RequirePackage{enumerate}
\RequirePackage{lmodern}
\RequirePackage{mathrsfs}
\RequirePackage{mleftright} % corrects spacing after \left( and before \right), etc
	\mleftright
\RequirePackage{booktabs}

%\RequirePackage{imakeidx} %* for indexing (load before hyperref)

%============Caminhos da Pasta das Figuras==============
\graphicspath{{figuras/}}
%# Capa
\usepackage{fancyhdr}
\fancypagestyle{firststyle}
{
   \fancyhf{}
\setlength{\headsep}{0.5cm}
\setlength{\headheight}{3cm}
\renewcommand{\headrulewidth}{0pt}
\lhead{\includegraphics*[width=\textwidth]{figc.PNG}}
}

\hypersetup{
  colorlinks = true,
  linkcolor=blue,   % color of internal links
  citecolor=blue,   % color of links to bibliography
  urlcolor=blue,    % color of external links
  bookmarksopen=true,
  pdfdisplaydoctitle=true
}

% comando para incluir o termo Figura antes da referencia
\newcommand{\reff}[1]{Figura (\ref{#1})}
% comando para incluir o termo Tabela antes da referencia
\newcommand{\reft}[1]{Tabela (\ref{#1})}


%==============================================================================
% Margens do texto
%==============================================================================
\newlength{\automatextcc@topmargin}\setlength{\automatextcc@topmargin}{30mm}
\newlength{\automatextcc@pagenummargin}\setlength{\automatextcc@pagenummargin}{20mm}
\newlength{\automatextcc@bottommargin}\setlength{\automatextcc@bottommargin}{20mm}
\newlength{\automatextcc@sidemargin}\setlength{\automatextcc@sidemargin}{30mm}
\setlength{\marginparwidth}{25mm}

% ajuste das medidas verticais
\setlength{\topmargin}{\automatextcc@pagenummargin}
        \addtolength{\topmargin}{-1em} % isto é +- a altura do núm. da página
        \addtolength{\topmargin}{-1in}
\setlength{\headheight}{1em}
\setlength{\headsep}{\automatextcc@topmargin}
        \addtolength{\headsep}{-\automatextcc@pagenummargin}
\setlength{\textheight}{\paperheight}
        \addtolength{\textheight}{-\automatextcc@topmargin}
        \addtolength{\textheight}{-\automatextcc@bottommargin}
\setlength{\footskip}{\automatextcc@bottommargin}

% ajuste das medidas horizontais
\setlength{\oddsidemargin}{\automatextcc@sidemargin}
        \addtolength{\oddsidemargin}{-1in}
\setlength{\evensidemargin}{\oddsidemargin}
\setlength{\textwidth}{\paperwidth}
        \addtolength{\textwidth}{-2\automatextcc@sidemargin}
\setlength{\marginparsep}{\automatextcc@sidemargin}
        \addtolength{\marginparsep}{-\marginparwidth}
        \setlength{\marginparsep}{.5\marginparsep}

%============================================================================================================
% Numeração das páginas
%============================================================================================================
\def\ps@automatextcc{
        \let\@oddfoot\@empty
        \let\@evenfoot\@empty
        \def\@evenhead{{\small\thepage}}
        \def\@oddhead{\hspace*{\fill}{\small\thepage}}
}
\pagestyle{empty}
\let\ps@plain\ps@empty
%============================================================================================================
% Definição do Autor, Orientador, Departamento, Universidade, tipo do trabalho(TCC) e data
%============================================================================================================
%autor
\renewcommand{\author}[1]{\gdef \@author{#1}}
%orientador
\newcommand{\advisor}[2]{
\gdef \@advisor{#1}
\gdef \@advisorgender{#2}
}
%informação do orientador:
\newcommand{\advisorinfo}[2]{\gdef \@advisorinfo{#1}
	\gdef \@advisoruni{#2}}
%banca examinadora e informação da banca:
\newcommand{\examinera}[1]{\gdef \@examinera{#1}}
\newcommand{\examinerainfo}[2]{\gdef \@examinerainfo{#1}
	\gdef \@examinerauni{#2}}
\newcommand{\examinerb}[1]{\gdef \@examinerb{#1}}
\newcommand{\examinerbinfo}[2]{\gdef \@examinerbinfo{#1}
	\gdef \@examinerbuni{#2}}










%universidade:
\def\ufrgs{Universidade Federal do Rio Grande do Sul}
\def\ee{Instituto de Matem{\'a}tica e Estat{\'i}stica}
%departamento:
\newcommand{\dept}[1]{
\gdef\@dept{#1}
}
%possibilidades:
\def\dest{Departamento de Estat{\'i}stica}
\def\DEST{\dest}

%tipo de trabalho:
\def\tcc{Trabalho de Conclus{\~a}o de Curso em Estat{\'i}stica}

\renewcommand{\date}[1]{\gdef\@date{#1}}
\newcommand{\location}{Porto~Alegre}







%%=========================================================================
%
%%Para Aparecer Referências Bibliográficas no Sumário
%
%%=========================================================================
%
%\def\thebibliography#1{\chapter*{\uppercase{\bibname}}\list{[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
%\advance\leftmargin\labelsep \advance\leftmargin -1em
%\advance\labelsep 2em
%\usecounter{enumi}}
%\sloppy\clubpenalty4000\widowpenalty4000
%\sfcode`\.=1000\relax\addcontentsline{toc}{chapter}{{{\bf\bibname}}}}
%\let\endthebibliography=\endlist
%\def\@lbibitem[#1]#2{\item[\@biblabel{#1}]\hspace*{-2em}\if@filesw
%      { \def\protect##1{\string ##1\space}\immediate
%        \write\@auxout{\string\bibcite{#2}{#1}}}\fi\ignorespaces}
%% impressao de capitulos no texto -- com numero (\chapter)
%\def\@makechapterhead#1{{\parindent 0pt \raggedright \large \bf \@hangfrom
%\par
% \nobreak \vskip \baselineskip} }
%
%













%==============================================================================
% Definição das palavras-chave (máx. 10)
%==============================================================================
\newcounter{kwcounter}
\newcommand{\keyword}[1]{
        \addtocounter{kwcounter}{1}
        \expandafter\gdef\csname keyw\alph{kwcounter}\endcsname{#1}
}
\newcommand{\@abstractkw}{%
        \@ifundefined{keywa}{}{\expandafter\MakeUppercase\keywa}%
        \@ifundefined{keywb}{}{, \keywb}%
        \@ifundefined{keywc}{}{, \keywc}%
        \@ifundefined{keywd}{}{, \keywd}%
        \@ifundefined{keywe}{}{, \keywe}%
        \@ifundefined{keywf}{}{, \keywf}%
        \@ifundefined{keywg}{}{, \keywg}%
        \@ifundefined{keywh}{}{, \keywh}%
        \@ifundefined{keywi}{}{, \keywi}%
        \@ifundefined{keywj}{}{, \keywj}%
}
\newcounter{kwcountere}
\newcommand{\keyworde}[1]{
        \addtocounter{kwcountere}{1}
        \expandafter\gdef\csname keywe\alph{kwcountere}\endcsname{#1}
}
\newcommand{\@abstractkwe}{%
        \@ifundefined{keywea}{}{\expandafter\MakeUppercase\keywea}%
        \@ifundefined{keyweb}{}{, \keyweb}%
        \@ifundefined{keywec}{}{, \keywec}%
        \@ifundefined{keywed}{}{, \keywed}%
        \@ifundefined{keywee}{}{, \keywee}%
        \@ifundefined{keywef}{}{, \keywef}%
        \@ifundefined{keyweg}{}{, \keyweg}%
        \@ifundefined{keyweh}{}{, \keyweh}%
        \@ifundefined{keywei}{}{, \keywei}%
        \@ifundefined{keywej}{}{, \keywesj}%
}
%==============================================================================
% Ajuste do sumário (nada indentado)
%==============================================================================
\newlength{\automatextcc@chapnumindent}\setlength{\automatextcc@chapnumindent}{1.8em}
%\renewcommand*{\l@chapter}[2]{
%        \addpenalty{-\@highpenalty}
%        \vskip -1ex \@plus\p@
%        \setlength\@tempdima{1.5em}
%        \begingroup
%        \parindent \z@ \rightskip \@pnumwidth
%        \parfillskip -\@pnumwidth
%        \leavevmode\bfseries\sffamily
%        \advance\leftskip\@tempdima
%        \hskip -\leftskip
%      esta é a definicao original da classe report:
%%      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
%        \@dottedtocline{0}{0em}{\automatextcc@chapnumindent}
%{\uppercase{#1}}{#2}
%%        \@dottedtocline{0}{0em}{\deletex@chapnumindent}
%        %{\MakeUppercase{#1}}{#2}
%        \penalty\@highpenalty
%        \endgroup
%}
\renewcommand*{\l@section}[2]
              {\@dottedtocline{1}{0em}{2.8em}{\textbf{#1}}{\hss #2}}
\renewcommand*{\l@subsection}{\@dottedtocline{2}{0em}{3.7em}}
\newif\if@aftertoc
\@aftertocfalse
%\renewcommand{\tableofcontents}{%
%        \chapter*{\contentsname}
%        \@starttoc{toc}\@aftertoctrue%
%}
%==============================================================================
% Definição das listas de abreviaturas e siglas e de símbolos
%==============================================================================
\newenvironment{listofabbrv}[1]{
        \chapter*{Lista de Abreviaturas}
        \begin{list}{\textbf{??}}{
                \settowidth{\labelwidth}{#1}
                \setlength{\labelsep}{1em}
                \setlength{\itemindent}{0mm}
                \setlength{\leftmargin}{\labelwidth}
                \addtolength{\leftmargin}{\labelsep}
                \setlength{\rightmargin}{0mm}
                \setlength{\itemsep}{.1\baselineskip}
                \renewcommand{\makelabel}[1]{\makebox[\labelwidth][l]{##1}}
        }
}{
        \end{list}
}
\newenvironment{listofsymbols}[1]{
        \chapter*{Lista de Símbolos}
        \begin{list}{\textbf{??}}{
                \settowidth{\labelwidth}{#1}
                \setlength{\labelsep}{1em}
                \setlength{\itemindent}{0mm}
                \setlength{\leftmargin}{\labelwidth}
                \addtolength{\leftmargin}{\labelsep}
                \setlength{\rightmargin}{0mm}
                \setlength{\itemsep}{.1\baselineskip}
                \renewcommand{\makelabel}[1]{\makebox[\labelwidth][l]{##1}}
        }
}{
        \end{list}
}
%==============================================================================
% Resumo (abstract) e Abstract (englishabstract)
%==============================================================================
\renewenvironment{abstract}{%
        \chapter*{Resumo}
}{%
        \par\vspace{2ex} %\vfill
        \noindent\textbf{Palavras-Chave:}{ \@abstractkw.}
}
\newenvironment{englishabstract}{%
        \chapter*{Abstract}
}{%
        \par\vspace{2ex} %\vfill
        \noindent\textbf{Keywords:}{ \@abstractkwe.}
}



%==============================================================================
% Capa
%==============================================================================

\newcommand{\maketitulo}{
\thispagestyle{firststyle}

\begin{center}
{\Large
\vspace*{\fill}
{Trabalho de Conclusão de Curso}

\vspace{8mm}

\large{\Large\textbf{\@title}}

\vspace{5cm}
}
\large{\@author}

\vspace*{\fill}

\today{}
\vspace{3cm}
\end{center}
\newpage
}





%==============================================================================
% Folha de Rosto
%==============================================================================
\newcommand{\makefolhaderosto}{%
	\begin{center}
		\textbf{\@author} \\[4cm]
		\large{\textbf{\@title}}\\[4cm]
	\end{center}
	\begin{flushright}
		\begin{minipage}{0.5\textwidth}
	Trabalho de Conclus\~ao apresentado \`a comiss\~ao de Gradua\c{c}\~ao do Departamento de Estat{\'i}stica da \ufrgs, como parte dos requisitos para obten\c{c}\~ao do t\'itulo de Bacharel em Estat{\'i}stica.\\[4cm]
	\@advisorgender:~\@advisor
		\end{minipage}
	\end{flushright}
	\vfill
	\begin{center}
		\location \\
		\@date
	\end{center}
	\newpage
}


%==============================================================================
%Folha de Aprovação 1
%==============================================================================
\newcommand{\makefolhadeaprovacaoA}{%
	%folha de aprovação:
	\begin{center}
		\textbf{\@author} \\[4cm]
		\large{\textbf{\@title}}\\[4cm]
	\end{center}
	\begin{flushright}
		\begin{minipage}{0.5\textwidth}%cuidar o pela e pelo aqui, não consegui fazer testes...
	Este Trabalho foi julgado adequado para obten\c{c}\~ao dos cr\'editos da disciplina Trabalho de Conclus\~ao de Curso em Estat{\'i}stica e aprovado em sua forma final pela \@advisorgender~ e pela Banca Examinadora.\\[0.5cm]
		\end{minipage}
		\begin{minipage}{0.6\textwidth}
	\@advisorgender:\hrulefill\\
	\@advisor, \@advisoruni\\
	\@advisorinfo\\[0.5cm]
		\end{minipage}
	\end{flushright}
\begin{flushleft}
Banca Examinadora:\\[0.5cm]
	\hspace{1cm} \@examinera, \@examinerauni\\
	\hspace{1cm}	\@examinerainfo\\[0.5cm]
%	\hspace{1cm} \@examinerb, \@examinerbuni\\
%	\hspace{1cm}	\@examinerbinfo\\[0.5cm]
\end{flushleft}
\vfill
\begin{center}
		\location \\
		\@date
\end{center}
}




%==============================================================================
%Folha de Aprovação 1
%==============================================================================
\newcommand{\makefolhadeaprovacaoB}{%
	%folha de aprovação:
	\begin{center}
		\textbf{\@author} \\[4cm]
		\large{\textbf{\@title}}\\[4cm]
	\end{center}
	\begin{flushright}
		\begin{minipage}{0.5\textwidth}%cuidar o pela e pelo aqui, não consegui fazer testes...
	Este Trabalho foi julgado adequado para obten\c{c}\~ao dos cr\'editos da disciplina Trabalho de Conclus\~ao de Curso em Estat{\'i}stica e aprovado em sua forma final pela \@advisorgender~ e pela Banca Examinadora.\\[0.5cm]
		\end{minipage}
		\begin{minipage}{0.6\textwidth}
	\@advisorgender:\hrulefill\\
	\@advisor, \@advisoruni\\
	\@advisorinfo\\[0.5cm]
		\end{minipage}
	\end{flushright}
\begin{flushleft}
Banca Examinadora:\\[0.5cm]
	\hspace{1cm} \@examinera, \@examinerauni\\
	\hspace{1cm}	\@examinerainfo\\[0.5cm]
	\hspace{1cm} \@examinerb, \@examinerbuni\\
	\hspace{1cm}	\@examinerbinfo\\[0.5cm]
\end{flushleft}
\vfill
\begin{center}
		\location \\
		\@date
\end{center}
}





%==============================================================================
% Formatação dos Capítulos Maiusculas
%==============================================================================
%\newcommand{\automatextcc@chapterfont}{\@setfontsize\Large{16}{20}}
%\def\@chapapp{}
%\def\@chapter[#1]#2{%
%        \refstepcounter{chapter}%
%        \addcontentsline{toc}{chapter}{\protect\numberline{\@chapapp\thechapter}#1}%
%%\if@relnum
%        \addtocontents{lof}{\protect\addvspace{10\p@}}%
%        \addtocontents{lot}{\protect\addvspace{10\p@}}%
%%\fi
%        \pagestyle{automatextcc}\let\ps@plain\ps@automatextcc%
%        \@makechapterhead{#2}\@afterheading
%}
%\def\@makechapterhead#1{%
%        \vspace*{90pt}
%        \reset@font\noindent{\automatextcc@chapterfont\sffamily\textbf{\@chapapp\thechapter\hspace{1em}\MakeUppercase{#1}}\par}
%        \addvspace{42pt}
%}
%\def\@schapter#1{%
%        \if@aftertoc\addcontentsline{toc}{chapter}{#1}\fi%
%%\if@relnum
%        \addtocontents{lof}{\protect\addvspace{10\p@}}%
%        \addtocontents{lot}{\protect\addvspace{10\p@}}%
%%\fi
%        \@makeschapterhead{#1}\@afterheading
%}


%==============================================================================
% Formatação dos Capítulos Normais
%==============================================================================
\newcommand{\automatextcc@chapterfont}{\@setfontsize\Large{16}{20}}
\def\@chapapp{}
\def\@chapter[#1]#2{%
        \refstepcounter{chapter}%
        \addcontentsline{toc}{chapter}{\protect\numberline{\@chapapp\thechapter}#1}%
%\if@relnum
        \addtocontents{lof}{\protect\addvspace{10\p@}}%
        \addtocontents{lot}{\protect\addvspace{10\p@}}%
%\fi
        \pagestyle{automatextcc}\let\ps@plain\ps@automatextcc%
        \@makechapterhead{#2}\@afterheading
}
\def\@makechapterhead#1{%
        \vspace*{90pt}
        \reset@font\noindent{\automatextcc@chapterfont\LARGE\textbf{\@chapapp\thechapter\hspace{1em}{#1}}\par}
        \addvspace{42pt}
}
\def\@schapter#1{%
        \if@aftertoc\addcontentsline{toc}{chapter}{#1}\fi%
%\if@relnum
        \addtocontents{lof}{\protect\addvspace{10\p@}}%
        \addtocontents{lot}{\protect\addvspace{10\p@}}%
%\fi
        \@makeschapterhead{#1}\@afterheading
}







%%%%%%%%%%%%%%%%%%% TITULOS EM MAIUSCULAS

%\def\@makeschapterhead#1{%
%        \vspace*{90pt}
%        \reset@font\centerline{\automatextcc@chapterfont\sffamily\textbf{\MakeUppercase{#1}}}
%        \addvspace{42pt}
%}

%%%%%%%%%%%%%%%%%%% TITULOS NORMAIS
\def\@makeschapterhead#1{%
        \vspace*{90pt}
        \reset@font\centerline{\automatextcc@chapterfont\LARGE\textbf{#1}}
        \addvspace{42pt}
}


\renewcommand{\appendix}[1][\appendixname]{%
        \setcounter{chapter}{0}%
        \renewcommand{\thechapter}{\Alph{chapter}}%
        \setcounter{tocdepth}{0}%
        \@captionlistingfalse%
        \gdef\@chapapp{\MakeUppercase{#1}\space}%
        \addtocontents{toc}{%
          \settowidth{\automatextcc@chapnumindent}{{\sffamily\textbf{\@chapapp}}}
          \addtolength{\automatextcc@chapnumindent}{1.8em}
        }%
}
\newcommand{\annex}{\appendix[\annexname]}
%==============================================================================
% Redefinição do conteúdo das listas de figuras e tabelas
%==============================================================================
\newif\if@captionlisting\@captionlistingtrue
\long\def\@caption#1[#2]#3{%
  \par
  \if@captionlisting%
    \addcontentsline{\csname ext@#1\endcsname}{#1}%
        {\protect\numberline{\csname fnum@#1\endcsname:}{\ignorespaces #2}}%
  \fi%
  \begingroup
    \@parboxrestore
    \if@minipage
      \@setminipage
    \fi
    \normalsize
    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup%
}
\renewcommand*{\l@figure}{\@dottedtocline{1}{0em}{5.5em}}
\let\l@table\l@figure

%longtable support
\RequirePackage{longtable}
\def\LT@c@ption#1[#2]#3{%
  \LT@makecaption#1\fnum@table{#3}%
  \def\@tempa{#2}%
  \ifx\@tempa\@empty\else
     {\let\\\space
     \addcontentsline{lot}{table}{\protect\numberline{\tablename~\thetable:}{#2}}}%
  \fi}

%%==============================================================================
%% Ajuste do formato das citações e referências
%%==============================================================================
%\setlength{\bibhang}{0pt}
%\let\cite\citep
%% Segundo a NBR-6023 as referencias devem ser alinhadas somente a esquerda
%\let\oldbibliography\bibliography
%\renewcommand{\bibliography}[1]{\begin{flushleft}\oldbibliography{#1}\end{flushleft}}

\renewcommand{\contentsname}{Sumário}