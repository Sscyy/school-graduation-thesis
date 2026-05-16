\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{sufethesis}

%% import global options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\ifchapterx\chapterxfalse

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax %revised by Quansheng Guan \ProcessOptions
\LoadClass[12pt,a4paper,openany]{book}

\newcommand{\thesissubject}{上海财经大学硕士学位论文}
\renewcommand{\today}{\the\year~年~\the\month~月~\the\day~日}


\RequirePackage{xcolor}
\RequirePackage{makeidx} % 索引
\RequirePackage{amsmath,amsthm,amsfonts,amssymb,bm,mathrsfs}%this package conflicts with xeCJK, place it before xeCJK to avoid the coflict.

%\RequirePackage[amsmath,thmmarks]{ntheorem}
\RequirePackage[final]{pdfpages}
%\usepackage{fontspec}
\usepackage{color}
\usepackage[unicode,
pdfstartview=FitH,
bookmarksnumbered=true,
bookmarksopen=true,
colorlinks=true,
pdfborder={0 0 1},
citecolor=blue,
linkcolor=black,
anchorcolor=green,
urlcolor=blue,
breaklinks=true
]{hyperref}

%% 设置页面布局
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[left=3.2cm,right=3.2cm,bottom=3.8cm,top=3.8cm,%设置页面上下左右各为25mm
 headheight=3cm,%页眉所占高度
 footskip=3cm %页脚所占高度
]{geometry}

%% 字体设置
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{ctex} % xeCJK package
\usepackage{CJKnumb}



%%%%%%%%%%%%% 设置行距
\renewcommand{\baselinestretch}{1.63}


\hbadness=10000 \tolerance=10000 \hfuzz=150pt % 去掉表格中的badboxes报警,看着不爽

% My fonts for sufe thesis
\renewcommand{\normalsize}{\zihao{-4}\songti\normalfont} % Normal font size 小四号宋体
\newcommand{\absnamecn}{\heiti\zihao{-2}}         % 中文“摘要”字样的字体
\newcommand{\abskeycn}{\heiti\zihao{-4}}       % 中文摘要“关键字”字样的字体
\newcommand{\abskeyscn}{\songti\zihao{-4}\normalfont} % 中文摘要关键字的字体
\newcommand{\absnameen}{\heiti\zihao{-4}}               % 英文``Abstract''字样的字体
\newcommand{\abskeyen}{\bfseries\zihao{-4}}   % 英文``KEY WORDS''字样的字体
\newcommand{\abskeysen}{\songti\zihao{-4}\normalfont}        % 英文关键字的字体
\newcommand{\headfont}{\songti\zihao{5}\normalfont} 	% 页眉字体
\newcommand{\toctitlefont}{\heiti\zihao{-2}}            % “目录”字样的字体
\newcommand{\tocchapterfont}{\heiti\zihao{-4}} % 目录上第X章的字体
\newcommand{\tocsectionfont}{\normalsize}         % 目录上X.Y节的字体
\newcommand{\tocsubsectionfont}{\normalsize}      % 目录上X.Y.Z小节的字体
\newcommand{\tocsubsubsectionfont}{\normalsize}      % 目录上X.Y.Z小节的字体
\newcommand{\textchapterfont}{\centering\heiti\zihao{-2}} % 正文上第X章的字体
\newcommand{\textsectionfont}{\centering\heiti\zihao{-3}}            % 正文上X.Y节的字体
\newcommand{\textsubsectionfont}{\heiti\zihao{4}}     % 正文上X.Y.Z小节的字体
\newcommand{\textsubsubsectionfont}{\heiti\zihao{-4}}     % 正文上X.Y.Z.K小节的字体
\newcommand{\footnotefont}{\songti\zihao{-5}\normalfont}     % 脚注字体
\newcommand{\ftcaptionfont}{\kaishu\zihao{5}\normalfont}         % 图表标题的字体
\newcommand{\reftitlefont}{\heiti\zihao{-2}}      % “参考文献”字样的字体
\newcommand{\refbodyfont}{\songti\zihao{5}\normalfont}          % 参考文献字体
\newcommand{\thanktitlefont}{\heiti\zihao{3}}    % “致谢”字样的字体
\newcommand{\appendixtitlefont}{\heiti\zihao{3}} % “附录”字样的字体




%% 图表环境
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%图题(由图号和图名组成)。图号为“图1-1”格式。图题置于图下，有图注或其他说明时应置于图题之上。
%表序与表名置于表上。
%图题和表题在lyx中插图或插表时，可以参照标题的位置，在上还是在下。

\RequirePackage{graphicx}
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png} % 如果插入的图片没有指定扩展名,那么依次搜索下面的扩展名所对应的文件
\RequirePackage{subfig} % config兼容subfigure命令
\RequirePackage{float} % 可以使用［H］命令
%\RequirePackage{ccaption}  %与subfig冲突，弃用
\RequirePackage{caption} % 定义图的标题格式：居中. 使用caption3.0
\DeclareCaptionFont{capFont}{\ftcaptionfont} % 表格名及图名
\DeclareCaptionLabelSeparator{twospace}{~~}
\captionsetup{ labelsep=twospace,% 去掉图标签后的冒号
  belowskip=0bp,aboveskip=0bp,
  font={capFont}, figurename=图,tablename=表,listfigurename=插图目录,listtablename=表格目录}
\captionsetup[figure]{position=bottom}
\captionsetup[subfloat]{captionskip=6bp,nearskip=0bp,farskip=0bp,topadjust=0bp,justification=centering}
%%%%%%%%%%定义双标题命令
\newcommand{\FigureBicaption}[2]{
  \renewcommand{\figurename}{图}
  \vspace{8pt}
  \caption{#1}
  \addtocounter{figure}{-1}
  \renewcommand{\figurename}{Fig.}
  \captionsetup{list=false}
  \vspace{6pt}
  \caption{#2}
  \captionsetup{list=true}
  \renewcommand{\figurename}{图}
}
\newcommand{\SubfloatCaption}[4]  % Two Parameters, the first one is the width of the subfigure,
{
  \subfloat[#2]{
    \label{#3}
    \addtocounter{subfigure}{-1} 
    \subfloat[\vspace{-5pt}#1]{#4}
  }%
}

%% 表格 
\RequirePackage{array} % 扩展了 array 和 tabular 环境功能
\RequirePackage{booktabs} % 增加对三线表格的支持 % \toprule，\midrule，\bottomrule
\usepackage{longtable} % 长表格支持，截断跨页
\RequirePackage{ctable, threeparttable} % 支持表格注释,复杂表格，一般用不上
%\RequirePackage{tabularx}
\captionsetup[table]{position=top}
%定义双标题命令
\newcommand{\TableBicaption}[2]{
  \renewcommand{\tablename}{表}
  \vspace{8pt}
  \caption{#1}
  \vspace{6pt}
  \addtocounter{table}{-1}
  \renewcommand{\tablename}{Table}
  \captionsetup{list=false}
  \caption{#2}
  \captionsetup{list=true}
  \renewcommand{\tablename}{表}
}

\renewcommand{\theequation}{\arabic{chapter}-\arabic{equation}}
\renewcommand{\thetable}{\arabic{chapter}-\arabic{table}}
\renewcommand{\thefigure}{\arabic{chapter}-\arabic{figure}}

\renewcommand{\thefootnote}{注\arabic{footnote}} \setcounter{footnote}{0}

\renewcommand\fps@figure{htbp} % 设置图浮动的默认参数
\renewcommand\fps@table{htbp}

\renewcommand{\textfraction}{0.07}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\floatpagefraction}{0.85}

%% 列表
\RequirePackage{enumitem}
\setlist{noitemsep,nolistsep,topsep=0pt,itemindent=1em}
\setenumerate[1]{label=(\arabic*), ref=\arabic*}
\setenumerate[2]{label=\arabic*),ref=\arabic*}
\setenumerate[3]{label=\roman*), ref=\theenumii.\roman*}
\setitemize{fullwidth,itemindent=\parindent,listparindent=\parindent,itemsep=0ex,partopsep=0pt,parsep=0ex}
\setenumerate{fullwidth,itemindent=\parindent,listparindent=\parindent,itemsep=0ex,partopsep=0pt,parsep=0ex}
\setdescription{fullwidth,itemindent=\parindent,listparindent=\parindent,itemsep=0ex,partopsep=0pt,parsep=0ex}


\RequirePackage[numbers,super,square,sort&compress]{natbib} % 修改引用参考文献格式
% 用natbib代替下面的一些设置
%\let\originalcite=\cite
%\renewcommand{\cite}[1]{\textsuperscript{\originalcite{#1}}}
%\renewcommand{\citet}[1]{\originalcite{#1}}

%% 程序代码格式
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[ruled,vlined,algochapter]{algorithm2e}
% algorithm2e 中文标题定制
\renewcommand{\algorithmcfname}{算法}
\renewcommand\thealgocf{\csname the\algocf@within\endcsname-\@arabic\c@algocf}%

\RequirePackage{listings}
\definecolor{mygray}{RGB}{245,245,245}
\lstset{
	language=matlab,
	tabsize=4, %
	frame=tb,
	commentstyle=\color{red!50!green!50!blue!50},
	rulesepcolor=\color{red!20!green!20!blue!20},%代码块边框为淡青色
	keywordstyle=\color{blue!90}\bfseries,
	%backgroundcolor=\color{mygray},
	showstringspaces=false,%不显示代码字符串中间的空格标记
	stringstyle=\ttfamily,
	basicstyle={\ttfamily},
	%breaklines=true, 
	keepspaces=true, %
	flexiblecolumns=true, %
	lineskip=-0.1pt,%行距
	fontadjust,
	captionpos=t,
	framextopmargin=1pt,framexbottommargin=1pt,abovecaptionskip=1pt,belowcaptionskip=1pt,
	%xleftmargin=4em,xrightmargin=4em, % 设定listing左右的空白
	extendedchars=false,columns=flexible,mathescape=false
	breakautoindent=true,
}
\DeclareCaptionFormat{mylst}{\hrule#1#2#3}
\captionsetup[lstlisting]{format=mylst,labelfont=bf,singlelinecheck=off,labelsep=space}
\renewcommand{\lstlistingname}{代码} %% 重命名Listings标题头 added by Guan


\RequirePackage{tcolorbox}
\tcbuselibrary{listings}
\newcommand{\tcblistingtitle}[1]{
\refstepcounter{lstlisting}
\zihao{5}\lstlistingname \thelstlisting~~ #1
} 

%参考文献bibi style
\RequirePackage[nottoc]{tocbibind} % 将参考文献、索引、表格目录和图形
\renewcommand{\bibname}{参考文献}

% 设置目录
%%%%%%%%%%%%%%%%%
\RequirePackage{titletoc} 
\renewcommand\contentsname{\centerline{\toctitlefont{目\quad{}录}}}
\titlecontents{chapter}[0ex]{\tocchapterfont}{%
    第\CJKnumber{\thecontentslabel}章\quad{}}{}{%
    \hspace{.5em}\titlerule*[5pt]{$\cdot$}\contentspage}%
\titlecontents{section}[1em]{\tocsectionfont}{%
    \thecontentslabel\quad{}}{}{%
    \hspace{.5em}\titlerule*[5pt]{$\cdot$}\contentspage}%
\titlecontents{subsection}[2em]{\tocsubsectionfont}{%
    \thecontentslabel\quad{}}{}{%
    \hspace{.5em}\titlerule*[5pt]{$\cdot$}\contentspage}%
\titlecontents{subsubsection}[3em]{\tocsubsubsectionfont}{%
    \thecontentslabel\quad{}}{}{%
    \hspace{.5em}\titlerule*[5pt]{$\cdot$}\contentspage}%

% 设定章节深度和目录深度
\setcounter{secnumdepth}{4}
\setcounter{tocdepth}{2}

% Text style
%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[pagestyles, rm]{titlesec}%调用titlesec涵盖fancyhdr的功能
\renewcommand{\chaptername}{第\CJKnumber{\thechapter}章}
\titleformat{\chapter}[hang]{\textchapterfont}{\textbf{第\CJKnumber{\thechapter}章}}{1em}\textbf{{}}
\titleformat{\section}[hang]{\textsectionfont}{第\CJKnumber{\arabic{section}}节}{1em}{}
\titleformat{\subsection}[hang]{\textsubsectionfont}{\thesubsection}{1em}{}
\titleformat{\subsubsection}[hang]{\textsubsectionfont}{\thesubsubsection}{1ex}{}% 小四黑体居左

%\titlespacing{\chapter}{0pt}{3ex plus .5ex minus .5ex}{3ex plus .5ex minus .5ex} % 缩短章节标题的上边距
\setlength{\parskip}{0ex}% 正文段前段后距：无空行
\titlespacing{\chapter}{0pt}{1pt}{1pt} % 缩短章标题上间距，revised by Quansheng Guan
\titlespacing{\section}{0pt}{1pt}{1pt}  % 标题间距
\titlespacing{\subsection}{0pt}{4pt}{0pt} 
\titlespacing{\subsubsection}{0pt}{6pt}{6pt} 

\newcommand{\linespacing}[1]{\linespread{#1}\selectfont}% 行距命令

%\sethead[ even-left ][ even-center ][ even-right ]    { odd-left }{ odd-center }{ odd-right }
\newpagestyle{revtitlestyle}{
 \ifchapterx 
  \sethead[][{\headfont{}\@chapterxname}][] % 页眉奇偶数页
  {}{{\headfont{}\thesissubject}}{} % 页眉奇数页
 \else
  \sethead[][{\headfont{}\chaptertitlename\quad\chaptertitle}][] % 页眉偶数页，五号字体，居中
  {}{{\headfont{}\chaptertitlename\quad\chaptertitle}}{} % 页眉奇数页，五号字体，居中
 \fi
  \setfoot[][\headfont{}\thepage][] %偶数页码五号宋体居右
   {}{\headfont{}\thepage}{}%页码五号宋体居右
  \headrule%上横线
  \setheadrule{1.2pt}%设置横线粗细
}

\let\orig@bibliography=\bibliography
\renewcommand{\bibliography}[1]{
	\cleardoublepage % 使参考文献使用 seunolabel 页面样式
	\refbodyfont
%        \chapterx% 改变页眉标题方式
        \chapterxname{\bibname}
	\let\ps@plain=\ps@revtitlestyle
	\pagestyle{revtitlestyle} % here and continue 
	\orig@bibliography{#1}
%	\normalsize
} 

\renewcommand\frontmatter{ %标题之后，开始中英文摘要，目录
    \pagenumbering{Roman}% Roman style page number
    \pagestyle{plain}
    \@mainmatterfalse
}%

\renewcommand\mainmatter{%
 \@mainmattertrue
  \chapterxfalse % 改变页眉标题方式
  \ifodd \arabic{page} % 目录后如果是奇数页结尾，则插入一个无页码空白页
   \clearpage
   \pagestyle{empty}
   \mbox{} %空白占位符
  \fi
  \clearpage
  \pagenumbering{arabic}

  \makeatletter
  \let\ps@plain=\ps@revtitlestyle
  \makeatother
  \pagestyle{revtitlestyle} % here and continue 
  % 开始正文部分
}
\renewcommand\backmatter{%
  % there is no label number for reference chapter
}
%==========
%\AtBeginDocument{\CJKindent{}%
%    \makeindex%
%    \normalsize%正文字体格式
%    \linespacing{1.63}% 主行距～1.5倍
%  }
%\AtEndDocument{}

%------------------

\def\SHUFET@label@classification{分类号}
\def\SHUFET@label@confidential{密级}
\def\SHUFET@label@UDC{UDC}
\def\SHUFET@label@serialnumber{编号}
\def\SHUFET@label@schoolnumber{学校代码}
\def\SHUFET@label@studentidnumber{学\hspace{2\ccwd}号}
\def\SHUFET@label@thesis{学位论文}
\def\SHUFET@label@advisor{指导教师：}
\def\SHUFET@label@degree{申请学位级别}
\def\SHUFET@label@major{论文类别：}
\def\SHUFET@label@submitdate{论文提交日期}
\def\SHUFET@label@defenddate{论文答辩日期}
\def\SHUFET@label@completedate{完成日期：}
\def\SHUFET@label@institute{培养单位}
\def\SHUFET@label@school{学位授予单位}
\def\SHUFET@label@chairman{答辩委员会主席}
\def\SHUFET@label@department{培养院系：}
\def\SHUFET@label@author{论文作者：}
%
% string values filled in the title page
%
\def\SHUFET@value@classification{}
\def\SHUFET@value@confidential{}
\def\SHUFET@value@UDC{}
\def\SHUFET@value@serialnumber{}
\def\SHUFET@value@schoolnumber{10272}
\def\SHUFET@value@studentidnumber{}
\def\SHUFET@value@school{上海财经大学}
\def\SHUFET@value@schoolEng{SHANGHAI ~UNIVERSITY~OF~FINANCE~AND~ECONOMICS}
\def\SHUFET@value@degreeEng{MASTER~~~~~DISSERTATION}
\def\SHUFET@value@degreeType{（专~业~硕~士）}
\def\SHUFET@value@title{~~~~~（论~文~题~目）~~~~~}
\def\SHUFET@value@titlemark{\SHUFET@value@title}
\def\SHUFET@value@author{（作~者~姓~名）}
\def\SHUFET@value@advisor{（姓名、专业技术职务、学位）}
\def\SHUFET@value@advisorinstitute{（单位）}
\def\SHUFET@value@major{}
\def\SHUFET@value@submitdate{}
\def\SHUFET@value@defenddate{}
\def\SHUFET@value@completedate()
\def\SHUFET@value@institute{上海财经大学}
\def\SHUFET@value@chairman{}
\def\SHUFET@value@department{}
\def\SHUFET@value@englishtitle{(English Title of Thesis)}


\def\SHUFET@underline[#1]#2{%
	\underline{\hbox to #1{\hfill#2\hfill}}}
\def\SHUFETunderline{\@ifnextchar[\SHUFET@underline\underline}
\newcommand\classification[1]{\def\SHUFET@value@classification{#1}}
\newcommand\confidential[1]{\def\SHUFET@value@confidential{#1}}
\newcommand\UDC[1]{\def\SHUFET@value@UDC{#1}}
\newcommand\serialnumber[1]{\def\SHUFET@value@serialnumber{#1}}
\newcommand\school[1]{\def\SHUFET@value@school{#1}}
\newcommand\department[1]{\def\SHUFET@value@department{#1}}
\newcommand\degree[1]{\def\SHUFET@value@degree{#1}}
\renewcommand\title[2][\SHUFET@value@title]{%
	\def\SHUFET@value@title{#2}
	\def\SHUFET@value@titlemark{\MakeUppercase{#1}}}
\newcommand\englishtitle[1]{\def\SHUFET@value@englishtitle{#1}}
\renewcommand\author[1]{\def\SHUFET@value@author{#1}}
\newcommand\advisor[1]{\def\SHUFET@value@advisor{#1}}
\newcommand\advisorinstitute[1]{\def\SHUFET@value@advisorinstitute{#1}}
\newcommand\major[1]{\def\SHUFET@value@major{#1}}
\newcommand\submitdate[1]{\def\SHUFET@value@submitdate{#1}}
\newcommand\defenddate[1]{\def\SHUFET@value@defenddate{#1}}
\newcommand\completedate[1]{\def\SHUFET@value@completedate{#1}}
%\newcommand\institute[1]{\def\SHUFET@value@institute{#1}}
\newcommand\chairman[1]{\def\SHUFET@value@chairman{#1}}
\newcommand\studentidnumber[1]{\def\SHUFET@value@studentidnumber{#1}}




\renewcommand\maketitle{%
	\cleardoublepage
	\thispagestyle{empty}
	\newgeometry{left=2.5cm,right=2.5cm}
	\begin{center}
		\bf\songti\zihao{-4}
		\begin{minipage}[p]{0.3\linewidth}
			\vspace{-2cm}
			\includegraphics[width=2.5cm]{figures/SHUFEBadge.jpg}
		\end{minipage}
		\hfill
		\begin{minipage}[p]{0.31\linewidth}
			\vspace{-2cm}
			\SHUFET@label@schoolnumber：\SHUFET@value@schoolnumber\\
			\SHUFET@label@studentidnumber：\SHUFET@value@studentidnumber
		\end{minipage}
		\vspace{0.5cm}
			\linespacing{1.2}{
			\begin{center}
			\includegraphics[width=9.5cm]{figures/SHUFEName.jpg}\\
			\zihao{4}\SHUFET@value@schoolEng\\
			\zihao{2}硕~~士~~学~~位~~论~~文\\
			\zihao{4}\SHUFET@value@degreeEng\\
			\kaishu\zihao{3}\SHUFET@value@degreeType
			 \end{center}}
		\vskip \stretch{1.5}
		\linespacing{1.2}{
		\bf\songti\zihao{2} 论文题目：\SHUFET@value@title
        }
		\vskip \stretch{3}
		\bf\songti\zihao{4}
		\def\tabcolsep{5pt}
		\def\arraystretch{1.5}
		\begin{tabular}{llcrl}
			\SHUFET@label@department &
			\multicolumn{4}{l}{\SHUFETunderline[200pt]{\SHUFET@value@department}}
			\\
			\SHUFET@label@major &
			\multicolumn{4}{l}{\SHUFETunderline[200pt]{\SHUFET@value@major}}
			\\
			\SHUFET@label@author &
			\multicolumn{4}{l}{\SHUFETunderline[200pt]{\SHUFET@value@author}}
			\\
			\SHUFET@label@advisor &
			\multicolumn{4}{l}{\SHUFETunderline[200pt]{\SHUFET@value@advisor}}
			\\
			\SHUFET@label@completedate &
			\multicolumn{4}{l}{\SHUFETunderline[200pt]{\SHUFET@value@completedate}}
		\end{tabular}
	\end{center}
\restoregeometry
\includepdf{figures/statement.pdf}  
\clearpage
}
%%摘要分中英文页，而且有两者排版方式：
%一种是采用\abstractCN和\abstractEN方式以“摘要”和“Abstract”为页标题
%另一种采用研究报告的标题为摘要页的标题，这时使用\charpter*{报告中文或英文标题}和\chapterx{摘要或Abstract}
\newenvironment{abstractCN}{
  \chapter*{\textbf{摘\quad 要}}
  \addcontentsline{toc}{chapter}{摘要} 
}{}

\newenvironment{abstractEN}
{
  \chapter*{\textbf{Abstract}}
  \addcontentsline{toc}{chapter}{Abstract}
}{}

\newcommand{\keywordsCN}[1]{
  \par
 \vspace{1em}
  \newcommand{\@keywords}{#1}
  {\noindent\abskeycn \textbf{关键词：}} {\abskeyscn \@keywords} % \noindent去掉缩进 
}
\newcommand{\keywordsEN}[1]{
  \par
 \vspace{1em}
  {\noindent\abskeyen Keywords:} {\abskeysen #1} % \noindent去掉缩进
}
%
\newcommand{\abstractx}[1]
{
 \begin{center}
  #1
 \par\end{center}
  \addcontentsline{toc}{chapter}{#1}
}

%有些章节名如“致谢”、“附录”，无需章节编号，相当于chapter*，但需要把章节名加入目录，相应的页眉也要改动
%通过\chapterx和\chapterxname来完成。
\def \chapterxname#1{\gdef\@chapterxname{#1}}
\newcommand{\chapterx}[1]
{
  \chapterxtrue
  \chapter*{\textbf{#1}}
  \chapterxname{\thesissubject}
  \addcontentsline{toc}{chapter}{#1}
}

\renewcommand{\appendix}[1]{%
  % \chaptertitle is not available for those chapters without label number.
   \cleardoublepage 
   \renewcommand{\thesection}{{#1}\arabic{section}} 
   \chapter*{\textbf{#1}}
   \chapterxname{\thesection\quad\sectiontitle}
   \addcontentsline{toc}{chapter}{#1}
   \setcounter{section}{0}
}

\renewenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\renewenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\renewenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%\vspace{-15pt}
                \item\relax}
               {\endlist}
               
 \renewenvironment{thebibliography}[1]
 {%\zihao{5}%
 	\chapterx{\bibname}%
 	\@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
 	\list{\@biblabel{\@arabic\c@enumiv}}%
 	{\settowidth\labelwidth{\@biblabel{#1}}%
 		\leftmargin\labelwidth
 		\advance\leftmargin\labelsep
 		\itemsep-4pt\topsep\z@\partopsep\z@\parskip\z@
 		\@openbib@code
 		\usecounter{enumiv}%
 		\let\p@enumiv\@empty
 		\renewcommand\theenumiv{\@arabic\c@enumiv}}%
 	\sloppy
 	\clubpenalty4000
 	\@clubpenalty \clubpenalty
 	\widowpenalty4000%
 	\sfcode`\.\@m}
 {\def\@noitemerr
 	{\@latex@warning{Empty `thebibliography' environment}}%
 	\endlist}
 
\newenvironment{publications}[1]
{\chapterx{发表文章目录}%
	\@mkboth{\MakeUppercase\SHUFET@label@publications}
	{\MakeUppercase\SHUFET@label@publications}%
	\list{\@biblabel{\@arabic\c@enumiv}}%
	{\settowidth\labelwidth{\@biblabel{#1}}%
		\leftmargin\labelwidth
		\advance\leftmargin\labelsep
		\@openbib@code
		\usecounter{enumiv}%
		\let\p@enumiv\@empty
		\renewcommand\theenumiv{\@arabic\c@enumiv}}%
	\sloppy
	\clubpenalty4000
	\@clubpenalty \clubpenalty
	\widowpenalty4000%
	\sfcode`\.\@m}
{\def\@noitemerr
	{\@latex@warning{Empty `publications' environment}}%
	\endlist}

\newenvironment{resume}
{\chapterx{个人简历}}
{}
\newenvironment{resumesection}[1]
{{\noindent \textbf{#1}}
	\list{}{\labelwidth\z@
		\leftmargin 2\ccwd}
	\item\relax}
{\endlist}
\renewenvironment{resumelist}[1]
{{\noindent \textbf{#1}}
	\list{}{\labelwidth\z@
		\leftmargin 4\ccwd
		\itemindent -2\ccwd
		\listparindent\itemindent}
	\item\relax}
{\endlist}      
\renewenvironment{thanks}
{\chapterx{致谢}}
{}         
 
       
            
%%定理环境
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%允许公式换页显示,否则大型推导公式都在一页内，
%一页显示不下放到第二页，导致很大的空白空间，很不好看
\allowdisplaybreaks[4]
%\setlength\jot{1pt} %公式之间的距离,TODO

\newtheoremstyle{definition}% name
  {0pt}%      Space above, empty = `usual value'
  {0pt}%      Space below
  {}% Body font \itshape
  {\parindent}%         Indent amount (empty = no indent, \parindent = para indent)
  {\bfseries}% Thm head font
  {:}%        Punctuation after thm head
  {0.5em}% Space after thm head: \newline = linebreak
  {}%         Thm head spec
\theoremstyle{definition}
\newtheorem{definition}{定义~}[chapter]
\newtheorem{example}{例~}[chapter]
\newtheorem{remark}{说明~}[chapter]


\newtheoremstyle{plain}% name
  {0pt}%      Space above, empty = `usual value'
  {0pt}%      Space below
  {\itshape}% Body font \itshape
  {\parindent}%         Indent amount (empty = no indent, \parindent = para indent)
  {\bfseries}% Thm head font
  {:}%        Punctuation after thm head
  {0.5em}% Space after thm head: \newline = linebreak
  {}%         Thm head spec
\theoremstyle{plain}
\newtheorem{proposition}{命题~}[chapter]
\newtheorem{lemma}{引理~}[chapter]
\newtheorem{theorem}{定理~}[chapter]
\newtheorem{axiom}{公理~}[chapter]
\newtheorem{corollary}{ 推论~}[chapter]
\newtheorem{case}{情形~}[chapter]
\newtheorem{conjecture}{猜想~}[chapter]
\newtheorem{property}{性质~}[chapter]
\newtheorem{assumption}{假设~}[chapter]

\renewenvironment{proof}{\vskip 1pt\indent \itshape 证明:~\normalfont}{\hfill$\square$\vskip 0.01\baselineskip} %$\blacksquare$
%\renewcommand{\proofname}{\hskip 2em 证明:~} % 
%%%%%%%%%%%%%%%%%   end of 定理环境
\setlength\abovedisplayskip{4pt}
\setlength\belowdisplayskip{4pt}



\endinput
%% End of file `sufethesis.cls'.
