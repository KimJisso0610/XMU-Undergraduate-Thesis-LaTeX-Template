# XMU 本科毕业论文模板

版次：第一版

作者：ReLU

最近一次更新时间：2022年6月30日

## 1. 前言

关于LaTeX的下载与安装，这里不做教学，请大家自行上网搜索。

本人的环境：TexLive + VisualStudio Code (推荐大家使用我这个)

(PS: 如果你在配环境这一步都没办法搞定，说明LaTeX不适合你，还是老老实实用Word吧，或者用OverLeaf)



然后给大家解析一下我的文件结构：

1. main.tex：所有项目文件的核心，负责将论文的各种部分串起来，务必先打开这个文件，接下来我也会详细说一下main.tex的代码；

2. thanks_abstract.tex：生成致谢与中英文摘要，然后在main.tex直接导入pdf文件，具体为什么要这样做，后面我也会说 (是因为技术太菜......)

3. content_en.tex：生成英文目录，然后在main.tex直接导入pdf文件；

4. appendix.tex：生成附录，然后在main.tex直接导入pdf文件；

5. frontmatter：存放正文前的项目，包括封面、诚信承诺书、致谢、摘要（中英文）、目录（中英文）

   1. cover.tex：生成封面；
   2. promise.tex：生成诚信承诺书；
   3. content_zh_cn.tex：生成中文目录；
   4. blank.pdf：空白页；

   其余都是没用的文件了

6. figure：存放图片
7. config：存放配置文件，例如页眉页脚设置、页码格式、字体字号等，非常庞大
   1. setting.tex：全局配置；
   2. frontmatter_config.tex：正文前的格式设置（页码必须为大写罗马数字）
   3. mainmatter_config.tex：正文的格式设置
   4. eng_content_config.tex：英文目录的格式设置

8. ch1~ch5：正文的每个章节
9. backmatter：存放正文后的项目，包括参考文献和附录
   1. reference.bib：参考文献列表
   2. appendix_a.tex / appendix_b.tex ：生成附录A和B



## 2. main.tex解读

main.tex是核心，读懂了就掌握了论文的总结构

```latex
\documentclass[UTF8]{book}

\renewcommand{\title}{模板示例：基于混沌学理论的数字图像加密技术}

\usepackage[fontset=none,heading=true]{ctex}
\setCJKmainfont{宋体}[AutoFakeBold={2.5}]
\setCJKsansfont{黑体}[AutoFakeBold={2.5}]
\setmainfont{Times New Roman}
\setsansfont{Arial}
\input{config/setting.tex}

\usepackage{titlesec}
\usepackage{setspace}


\begin{document}

    \frontmatter
    \input{config/mainmatter_config.tex}
    \input{frontmatter/cover.tex}
    \includepdf[pages=1]{frontmatter/blank.pdf}
    \pagenumbering{Roman}
    \input{frontmatter/promise.tex}
    \setcounter{page}{2}
    \addcontentsline{toc}{chapter}{致谢}
    \includepdf[pages=1]{thanks_abstract.pdf}
    \setcounter{page}{3}
    \addcontentsline{toc}{chapter}{摘要}
    \includepdf[pages=2]{thanks_abstract.pdf}
    \setcounter{page}{4}
    \addcontentsline{toc}{chapter}{Abstract}
    \includepdf[pages=3]{thanks_abstract.pdf}
    \setcounter{page}{5}
    \addcontentsline{toc}{chapter}{目录}
    \input{frontmatter/content_zh_cn.tex}
    \setcounter{page}{6}
    \addcontentsline{toc}{chapter}{Table of Contents}
    \includepdf[pages=6]{content_en.pdf}
    

    \input{config/mainmatter_config.tex}    
    \mainmatter
    \input{ch1/ch1.tex}
    \input{ch2/ch2.tex}
    \input{ch3/ch3.tex}
    \input{ch4/ch4.tex}
    \input{ch5/ch5.tex}
    %\backmatter
    
    \begin{center}
        \bibliography{backmatter/reference}
        \addcontentsline{toc}{chapter}{参考文献}
    \end{center}
    \includepdf[pages=1]{frontmatter/blank.pdf} % 参考文献是奇数页的话请使用！
    \addcontentsline{toc}{chapter}{附录\space A\quad 关于本论文模板的相关说明}
    \includepdf[pages=1]{appendix.pdf}
    \addcontentsline{toc}{chapter}{附录\space B\quad 附录代码示例}
    \includepdf[pages=2-5]{appendix.pdf}
    
\end{document}
```

首先，```\documentclass[UTF8]{book}```是论文的起始，book代表它的章节格式为book格式，参数带有UTF8表示里面含有UTF8格式的字符；

```\renewcommand{\title}{模板示例：基于混沌学理论的数字图像加密技术}```是一个命令重载，表示这篇论文的Title叫做“模板示例：基于混沌学理论的数字图像加密技术”；

```\usepackage[fontset=none,heading=true]{ctex}```表示使用ctex包，有两个可选参数，具体是什么意思，我忘了......

```\setCJKsansfont{黑体}[AutoFakeBold={2.5}]```和```\setCJKmainfont{宋体}[AutoFakeBold={2.5}]```表示中文字体族的设置，主字体族为宋体，无衬线字体族为黑体，使用AutoFakeBold表示使用假加粗（因为LaTex的主字体族加粗是变成衬线字体族），而使用假加粗则可以保证字体不变，并且在字体的本身加粗；

同样地，下面两句表示英文字体族的设置；

```\input{config/setting.tex}```表示导入setting.tex，相当于把setting.tex中的代码完全copy到main.tex的这个位置；

再下面就是使用一些别的包；

```\frontmatter```表示接下来是正文前的部分；

```\input{frontmatter/cover.tex}```表示将cover.tex的代码完全copy到这个位置，之后编译就可以生成封面；

```\includepdf[pages=1]{frontmatter/blank.pdf}```表示将blank.pdf的第一页完全镶嵌在这里（并且是新的一页），这样就能达到封面和诚信承诺书中间隔着一个空白页的效果；

```\pagenumbering{Roman}```表示接下来的页码使用大写罗马数字；

```\setcounter{page}{2}```表示将此页的页码设置为2；

```\addcontentsline{toc}{chapter}{致谢}```将此页与目录进行一个链接，是一个章，名字为“致谢”；

后续的内容重复，不再赘述；

```\mainmatter```表示接下来是正文部分；

```\bibliography{backmatter/reference}```表示此处要放参考文献；

后续的内容重复，不再赘述。



## 3. 编译方法

要按照步骤来！否则就会编译出错

1. 先编译thanks_abstract.tex / content_en.tex / appendix.tex，得到PDF，此处使用两次Recipe: XeLaTeX即可；
2. 再编译main.tex，此处需要使用Recipe: xelatex -> bibtex -> xelatex*2，最后得到main.pdf就是最后的论文文档；



## 4. 需要改进的地方

虽然最后编译出的论文像样了，但是估计还不能完全达到要求

1.  英文目录不可以直接跳转，并且中文目录的正文前部分也不能正确跳转；
2. 注释写得比较少。

未来会逐渐更新版本，以达到最好，也欢迎大家一起帮我修改！