# Generative Language Autoencoders Are Lossless Embedding Models 

```tex
\section{Introduction}

Generative Language Autoencoder: GLAE.

Use SFT for simple implementation: 

\begin{figure}[htbp]
\vspace{-2mm}
\centering
\begin{tcolorbox}[width=\textwidth,colback=gray!2!white,colframe=gray!50!blue]
\textbf{Input:} 
$$
\text{Encode this text into 1-word length embedding:} <text> [text] </text>
$$

\textbf{Response:} 

$$
<embedding> [embedding] </embedding>
$$
\textbf{Input:} 

$$
\text{Decode this embedding into text:} <embedding> [embedding] </embedding>
$$

\textbf{Response:} 
$$
<text> [text] </text>
$$
\end{tcolorbox}
\caption{SFT data for training Generative Language Autoencoder.}
\label{fig:method-GLAE}
\end{figure}

One can use Total Coding Rate (TCR)~\citep{ma2007segmentation,li2022neural} uniformity loss: 

\begin{equation}
    \mathcal{L}_{1,\text{TCR}} = \operatorname{TCR}(\mathbf{embedding}_{j\in\{1,\cdots,B\}}). 
\end{equation}

One can also use the repulsive part of InfoNCE contrastive loss~\citep{chen2020simple}: 

\begin{equation}
    \mathcal{L}_{1,\text{contrastive}} = -\log \frac{1}{\sum_{j=1}^B \exp \left(\operatorname{sim}\left(\mathbf{embedding}_1, \mathbf{embedding}_j\right) / \tau\right)}. 
\end{equation}

\begin{equation}
    \mathcal{L}_2 = -\frac{1}{N}\sum_{i=1}^N\log P_i(\text{text}_i | \mathbf{embedding}, \text{text}_{<i}). 
\end{equation}

Alternative optimization or simply add these two losses: 

\begin{equation}
    \mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2.
\end{equation}

We use the $-2$ layer representation as our embedding.
```

