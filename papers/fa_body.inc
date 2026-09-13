
\begin{abstract}
\noindent
A structure-conditioned diffusion adapter is trained for one operating point.
Canny edges, a depth map, a sketch: each fixes how tightly the output must
follow the condition, and moving that trade-off means training again. We train
a single adapter on conditions whose information content is randomised --- a
high-pass filter whose cutoff $r$ is drawn per sample --- and read $r$ back at
test time as a dial. On FFHQ-512, structure consistency runs from
\scMainZeroFive{} at $r{=}0.05$ to \scMainThreeZero{} at $r{=}0.3$ while
sample diversity rises from \divMainZeroFive{} to \divMainThreeZero{}, and FID
stays between \fidMainZeroFive{} and \fidMainOneZero{} over that span. We
report the sweep out to $r{=}1$ but claim the dial only on
$r \in [0, 0.3]$, above which the condition retains too little energy for
structure consistency to be interpretable. Turning the adapter's strength down is not the same knob. At matched structure
consistency --- scale 0.50 reaches \scHalfScale{} and the cutoff $r{=}0.3$
reaches \scMainThreeZero{} --- the strength setting costs FID \fidHalfScale{}
where the cutoff costs \fidMainThreeZero{}, so the cutoff dominates on both axes at once. Once an artist filters a hand drawing at $r$, the dial works on drawings too.

Two findings came out of building it. The frozen SD VAE destroys a high-pass
condition before any adapter can read it: only \vaeNativeThreeZero{} of the
condition survives an encode--decode round trip at $r{=}0.3$ and
\vaeNativeSevenZero{} at $r{=}0.7$. And between these two injection designs
placement matters more than width --- skip-only retains SC \pathSkipOnlyZero{}
against \pathAttnOnlyZero{} for attention-only, with both paths at
\pathBothZero{} --- while quadrupling the adapter's width inside the
attention-only design buys far less. The adapter trains
\ourParams{} parameters, small next to ControlNet but not next to recent lightweight control modules; we make no efficiency claim.
\CCSblock
\end{abstract}

\section{Introduction}

ControlNet~\cite{zhang2023adding} and its successors~\cite{mou2023adapter,zhao2023controlnet,qin2023unicontrol,li2024controlnet} solved conditional control and then froze it.
An adapter trained on Canny edges follows Canny edges; one trained on depth
follows depth. The strength of that following is a property of the trained
weights, so an artist who wants the same sketch honoured more loosely has to
pick a different checkpoint or accept whatever the one they have was trained to
do.

We change what the condition is rather than how hard the model listens to it.
The condition is the image after a radial high-pass filter with normalised
cutoff $r$. At $r{=}0$ the model receives the grayscale image and the task is
colourisation. As $r$ grows the filter removes more of the spectrum, until at
$r{=}1$ almost nothing is left: measured on FFHQ, a cutoff of $0.05$ already
discards 97.5\,\% of the spectral energy. Training draws $r$ per sample, so one
set of weights sees the whole range, and $r$ becomes an inference-time
parameter.

Our contributions are:

\begin{itemize}\itemsep2pt
\item A control dial. One adapter spans structure consistency
      \scMainZeroFive{} to \scMainThreeZero{} on $r \in [0.05, 0.3]$ with FID
      between \fidMainZeroFive{} and \fidMainOneZero{} (Sec.~\ref{sec:dial}).
\item Evidence that the dial is not a rescaled version of adapter strength.
      At the closest matched pair the two settings reach structure consistency
      \scHalfScale{} and \scMainThreeZero{}, at FID \fidHalfScale{} and
      \fidMainThreeZero{} (Sec.~\ref{sec:strength}).
\item A measurement of how much of a high-pass condition survives the frozen
      VAE, which explains why latent-space conditioning fails for sparse
      structure (Sec.~\ref{sec:vae}).
\item A controlled comparison of injection sites showing that placement
      matters more than adapter capacity or receptive field
      (Sec.~\ref{sec:where}).
\end{itemize}

\section{Related work}

\paragraph{Conditional control.} Text-to-image diffusion
\cite{ho2020denoising,song2020denoising,rombach2021high,nichol2021glide,saharia2022photorealistic}
became spatially controllable with ControlNet~\cite{zhang2023adding}, which
duplicates the encoder and injects residuals through zero convolutions.
T2I-Adapter~\cite{mou2023adapter} does the same job with a small feature
extractor, IP-Adapter~\cite{ye2023adapter} adds decoupled cross-attention for
image prompts, and GLIGEN~\cite{li2023gligen} grounds generation on boxes.
Uni-ControlNet~\cite{zhao2023controlnet} and UniControl~\cite{qin2023unicontrol}
fold many condition types into one model, and
ControlNet++~\cite{li2024controlnet} closes a consistency loop on the produced image.
A second line shrinks the control branch on the same UNet:
ControlNet-XS~\cite{zavadski2023controlnet} couples a much smaller control
network to encoder and decoder in both directions, CtrLoRA~\cite{xu2024ctrlora}
adapts a shared base ControlNet to each new condition with LoRA, and
Simple-ControlNet~\cite{feng2025simplifying} feeds the condition through a
single-scale projection at the top of the UNet with LoRA on self-attention, in
under 3M parameters, exposing a scale on that projection at inference. Our
adapter is not smaller than these. Each of these fixes the condition type and, with it, how tightly the
output must follow: the operating point is a property of the trained weights.
Our cutoff moves that point at inference on one set of weights.

\paragraph{Efficient control on transformer backbones.} As backbones moved from
UNets to diffusion transformers~\cite{peebles2022scalable,esser2024scaling},
control moved from duplicated encoders to lightweight modules.
ControlNeXt~\cite{peng2024controlnext} removes the parallel branch,
OminiControl~\cite{tan2024ominicontrol} reuses the backbone itself as the
condition processor, and EasyControl~\cite{zhang2025easycontrol} and
NanoControl~\cite{liu2025nanocontrol} inject conditions through
LoRA-style~\cite{hu2021lora} modules at a small fraction of the parameter cost.
Recent work also examines where conditions should enter:
Laytrol~\cite{huang2025laytrol} on preserving pretrained knowledge under layout
control, PixelControl~\cite{lin2026pixelcontrol} on condition fidelity, and
Zhou et al.~\cite{zhou2026rethinking} on removing redundant attention in multi-condition
DiTs. Our injection-site comparison (Sec.~\ref{sec:where}) reaches a compatible
conclusion on a UNet backbone: placement dominates capacity.

\paragraph{Frequency in diffusion.} Frequency decomposition appears in diffusion
mostly as an intervention on the model's own signals rather than on its
condition. FreeU~\cite{si2023freeu} rebalances skip and backbone features
spectrally at no training cost, FreSca~\cite{huang2025fresca} scales frequency
bands of the noise prediction and, like us, uses an energy-based cutoff,
DeCo~\cite{ma2025deco} decouples frequencies in pixel-space diffusion, and
FeRA~\cite{yin2025fera} routes adaptation by frequency energy.
FOCUS~\cite{tjio2025focus} conditions the reverse process on learned frequency
priors for test-time adaptation. Closest to us in spirit,
Prasain~\cite{prasain2026beyond} conditions on wavelet bands instead of edge maps for
map-to-satellite synthesis. None of them randomises the condition's
\emph{bandwidth} during training so that the band becomes a control at
inference. The general pattern --- randomise a conditioning parameter while
training, expose it as an inference knob --- is not itself new:
SDXL~\cite{podell2023sdxl} samples crop coordinates during data loading and
surfaces them as a user control. Our delta is the choice of parameter, and the
dissociation that follows from it: bandwidth and adapter strength are not
interchangeable knobs.

\paragraph{Structure preservation.} StructDiff~\cite{he2026structdiff} targets
structure-preserving spatially controllable generation, and sketch-conditioned
pipelines have long used edge maps as the structural carrier. The idea that a
frequency band is itself an editing axis, and that a drawing should act as a
soft constraint rather than a hard one, both predate diffusion in the graphics
literature; our motivating sentence borrows from that line and the bibliography
below does not yet reflect it. Our condition is
a continuum rather than a fixed extractor, which is what makes a dial possible.

\paragraph{Training and evaluation.} We train with classifier-free
guidance~\cite{ho2022classifier} dropout and Min-SNR loss
weighting~\cite{hang2023efficient}, and evaluate with FID~\cite{heusel2017gans}
under the clean-FID protocol~\cite{parmar2021aliased},
LPIPS~\cite{zhang2018unreasonable} and CLIP score~\cite{radford2021learning} on
FFHQ~\cite{karras2018style} with BLIP~\cite{li2022blip} captions. SDXL and
SD3~\cite{podell2023sdxl,esser2024scaling} are the natural next backbones.

\section{Method}

\paragraph{Condition.} Given an image $x$, we convert to grayscale, take the
2D FFT, zero every frequency inside a disc of radius $r \cdot \min(H,W)/2$,
invert, and min--max normalise to $[0,1]$. The normalisation is deliberate: it
removes the absolute scale of the filter response, so the model has to infer
structure density from spatial statistics. We hypothesise that this is what
lets a hand drawing, which carries no $r$, be used at test time; we do not
ablate the normalisation.

\input{fig_mask}

\paragraph{Condition encoder.} A strided convolutional stem reads the condition
image directly at $512^2$ and reduces it $8\times$. We do not encode the
condition with the frozen VAE; Sec.~\ref{sec:vae} shows why.

\paragraph{Injection.} Encoder features enter the UNet in two places. Zero-initialised
$1{\times}1$ heads produce one residual per decoder skip tensor and one for the
mid block, added through the UNet's existing ControlNet interface. In parallel,
each cross-attention site attends over a $3{\times}3$ window of the condition
features, gated by a zero-initialised per-channel vector. The window is not
what carries the effect: at $1{\times}1$ the softmax runs over a single position,
the site reduces to a gated value projection, and structure consistency is no
lower (Sec.~\ref{sec:where}). Both paths start as
exact no-ops: with them switched off the network reproduces stock
Stable Diffusion~1.5~\cite{rombach2021high} bit for bit, which we assert before every training run.

\input{fig_arch}

\paragraph{Training.} SD1.5~\cite{rombach2021high} stays frozen. We train
\ourParams{} parameters on FFHQ-512~\cite{karras2018style} with
BLIP~\cite{li2022blip} captions, split \nTrain{} train / \nValid{} validation /
\nTest{} test by a seeded shuffle of the 70,000 images (5\% test, 2\%
validation, seed 2026). AdamW, learning rate $10^{-4}$, $\beta=(0.9,0.999)$,
cosine schedule with 500 warm-up steps, gradient clipping at 1.0, batch 32,
bf16, 12 epochs on one H200. The loss is $\epsilon$-prediction with Min-SNR
weighting~\cite{hang2023efficient}, $\gamma=5$. Text conditioning is dropped
with probability 0.1 and the structure condition independently with probability
0.05. An EMA of the adapter is kept with a 10-step interval. $r$ is drawn
uniformly from $[0,1]$ per sample. Code and adapter weights accompany the
submission.

\section{Experiments}

\subsection{Metrics}
\label{sec:metrics}

We evaluate on 500 of the \nTest{} held-out images with a DDIM sampler at 50
steps, CFG 7.5, seed 2026.

\paragraph{Structure consistency.} SC converts the generated image to grayscale,
applies the \emph{same} high-pass filter at the \emph{same} cutoff the model was
given, and takes the Pearson correlation against the given condition over the
flattened pixel vector, per image, averaged over images. Both signals are the
per-image min--max normalised filter output, so the correlation is scale free.
Two anchors bound it: feeding the ground truth back gives exactly 1.000 at every
$r$, and an unrelated face gives 0.122 at $r{=}0$ and ${\approx}{-}0.02$ above
it. Because the instrument moves with $r$, a row-to-row difference mixes model
behaviour with instrument change, so we report a cutoff-independent metric
beside it.

\paragraph{Edge-F1.} Canny edges (thresholds 100 and 200 on the 8-bit grayscale
image) are taken from the generation and from the reference and matched with a
2-pixel tolerance, implemented as a $5{\times}5$ dilation of each map before
intersecting; F1 is computed per image and averaged. It does not depend on $r$,
which is why it appears in Table~\ref{tab:main}: \edgeMainZeroFive{} at
$r{=}0.05$ falling to \edgeMainOneHundred{} at $r{=}1$, so the dial survives a
cutoff-independent instrument, with a floor near \edgeMainOneHundred{} that
aligned FFHQ faces supply for free.

\paragraph{Diversity.} Mean pairwise LPIPS (AlexNet backbone) over $K{=}4$
samples generated from one condition with different seeds; higher means the
condition constrains less. It is not independent of the LPIPS column: across the
sweep LPIPS $-$ Diversity is \divGap{}\,$\pm$\,\divGapSd{}, nearly constant over
a range of 0.20, so the two carry close to the same information and we draw no
separate conclusion from Diversity alone.

\paragraph{FID and CLIP.} Clean-FID~\cite{heusel2017gans,parmar2021aliased}
between the 500 generations and the same 500 references, and CLIP score with
ViT-B/32~\cite{radford2021learning} against each image's own caption.

\begin{figure*}[t]
\centering
\begin{tikzpicture}
\node[inner sep=0, anchor=south west] (d) at (0,0) {\includegraphics[width=\textwidth]{figures/dial}};
\foreach \lab [count=\i from 0] in {reference, $r{=}0$, $r{=}0.05$, $r{=}0.1$, $r{=}0.2$, $r{=}0.3$, $r{=}0.5$, $r{=}0.7$, $r{=}1$}
  \node[font=\scriptsize, anchor=south, inner sep=1.5pt]
    at ($(d.north west)!{(\i*390+192)/3504}!(d.north east)$) {\lab};
\end{tikzpicture}
\caption{One seed and one caption per row; only $r$ changes. The rows are the
first three test images in generation order. As $r$ rises, background detail
thins and clothing and colour drift from the reference while the face layout
holds; at $r{=}1$ the pose of the second row changes as well. The reference is
never shown to the model.}
\label{fig:dial}
\end{figure*}

\subsection{The dial}
\label{sec:dial}

Table~\ref{tab:main} gives the operating curve. On $r \in [0.05, 0.3]$
structure consistency falls at every step and sample diversity rises at every
step, which is the trade-off the dial is meant to expose. Two qualifications
belong with that sentence. Going from $r{=}0$ to $r{=}0.05$ structure
consistency \emph{rises} by \riseDelta{} ($z = \riseZ{}$, paired within one
checkpoint), so the sweep is not monotone at its left end; the same rise appears
in Edge-F1 (\edgeMainZero{} to \edgeMainZeroFive{}), so it is model behaviour
rather than an artefact of the SC instrument. And at the right end $r{=}0.5$ and
$r{=}0.7$ are not separable across seeds (paired $t = \tailT{}$, $df=2$,
critical \tailCrit{}). FID stays between \fidMainZero{} and
\fidMainOneHundred{} across the entire sweep.

Figure~\ref{fig:dial} holds the seed and the caption fixed and varies only $r$.
Because the caption is derived from the reference, it could be carrying part of
the structure. We repeated the sweep with one generic caption, ``a photo of a
person'', for every image. Structure consistency barely moves ---
\genSCZero{}, \genSCOneZero{} and \genSCThreeZero{} at $r{=}0$, $0.1$ and $0.3$,
against \scMainZero{}, \scMainOneZero{} and \scMainThreeZero{} with captions ---
while FID rises to \genFidZero{}--\genFidThreeZero{}, as it should once the
caption stops describing the image. The structure comes from the condition.
Figure~\ref{fig:curve} plots the curve and, below it, both dials in the same
structure-consistency/FID plane.

\input{fig_curve}

\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrrr}
\toprule
$r$ & SC $\pm$ SEM & seed sd & Edge-F1 & Diversity & FID \\
\midrule
0.00 & \scMainZero\,$\pm$\,\semMainZero           & \sSdZero       & \edgeMainZero      & \divMainZero      & \fidMainZero \\
0.05 & \scMainZeroFive\,$\pm$\,\semMainZeroFive   & \sSdZeroFive   & \edgeMainZeroFive  & \divMainZeroFive  & \fidMainZeroFive \\
0.10 & \scMainOneZero\,$\pm$\,\semMainOneZero     & \sSdOneZero    & \edgeMainOneZero   & \divMainOneZero   & \fidMainOneZero \\
0.20 & \scMainTwoZero\,$\pm$\,\semMainTwoZero     & \sSdTwoZero    & \edgeMainTwoZero   & \divMainTwoZero   & \fidMainTwoZero \\
0.30 & \scMainThreeZero\,$\pm$\,\semMainThreeZero & \sSdThreeZero  & \edgeMainThreeZero & \divMainThreeZero & \fidMainThreeZero \\
\midrule
\multicolumn{6}{l}{\itshape beyond the range we claim} \\
0.50 & \scMainFiveZero\,$\pm$\,\semMainFiveZero   & \sSdFiveZero   & \edgeMainFiveZero  & \divMainFiveZero  & \fidMainFiveZero \\
0.70 & \scMainSevenZero\,$\pm$\,\semMainSevenZero & \sSdSevenZero  & \edgeMainSevenZero & \divMainSevenZero & \fidMainSevenZero \\
1.00 & \scMainOneHundred\,$\pm$\,\semMainOneHundred & \sSdOneHundred & \edgeMainOneHundred & \divMainOneHundred & \fidMainOneHundred \\
\bottomrule
\end{tabular}
\caption{Operating curve, 500 of the \nTest{} test images. SEM is over those
500 images; seed sd is over three training seeds. Edge-F1 is a cutoff-independent
reading of the same runs and is reported alongside SC because SC re-extracts at
the same cutoff the model was given (Sec.~\ref{sec:metrics}). FID has a single
estimate per cell and no variance; we use it for row-to-row comparison only.}
\label{tab:main}
\end{table}

Across seeds 2026, 2027 and 2028 the curve is stable: SC at $r{=}0.05$ is
\seedMeanZeroFive{}\,$\pm$\,\seedStdZeroFive{} and at $r{=}0.3$ it is
\seedMeanThreeZero{}\,$\pm$\,\seedStdThreeZero{}.

\subsection{Strength is not the same knob}
\label{sec:strength}
An obvious alternative is to train on one cutoff and turn the adapter's
contribution down at inference, the knob lightweight control modules
expose~\cite{feng2025simplifying}. We ran it on the adapter trained at fixed
$r{=}0.1$ (Sec.~\ref{sec:spec}), scaling both injection paths, and, as a second
check, scaled the randomised-$r$ model at the same cutoff.

\begin{table}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}
\toprule
Setting & SC & Diversity & FID \\
\midrule
\multicolumn{4}{l}{\itshape fixed-$r{=}0.1$ adapter, scaled} \\
scale 0.25 & \sscSCQuarter & \sscDivQuarter & \sscFidQuarter \\
scale 0.50 & \sscSCHalf    & \sscDivHalf    & \sscFidHalf \\
scale 0.75 & \sscSCThreeQ  & \sscDivThreeQ  & \sscFidThreeQ \\
scale 1.00 & \sscSCOne     & \sscDivOne     & \sscFidOne \\
\midrule
\multicolumn{4}{l}{\itshape randomised-$r$ adapter at $r{=}0.1$, scaled} \\
scale 0.25 & \scQuarterScale & \divQuarterScale & \fidQuarterScale \\
scale 0.50 & \scHalfScale    & \divHalfScale    & \fidHalfScale \\
scale 0.75 & \scThreeQScale  & \divThreeQScale  & \fidThreeQScale \\
\midrule
\multicolumn{4}{l}{\itshape randomised-$r$ adapter, cutoff moved} \\
$r{=}0.2$ & \scMainTwoZero   & \divMainTwoZero   & \fidMainTwoZero \\
$r{=}0.3$ & \scMainThreeZero & \divMainThreeZero & \fidMainThreeZero \\
\bottomrule
\end{tabular}
\caption{Strength against cutoff. The fixed-$r$ adapter at scale 0.50 reaches
SC \sscSCHalf{} at FID \sscFidHalf{}; moving the cutoff brackets that value with
$r{=}0.2$ (\scMainTwoZero{}, FID \fidMainTwoZero{}) and $r{=}0.3$
(\scMainThreeZero{}, FID \fidMainThreeZero{}), both at lower FID. Scaling the
randomised model behaves the same way.}
\label{tab:strength}
\end{table}

Both scaled curves leave the image distribution as they go. At scale 0.25 the
fixed-$r$ adapter reaches FID \sscFidQuarter{} and diversity \sscDivQuarter{},
beyond anything the cutoff produces over its whole sweep, where FID stays within
\fidMainZero{}--\fidMainOneHundred{} and diversity peaks at
\divMainOneHundred{}. Turning the adapter down removes structure by degrading
the sample; narrowing the condition's band removes it while image quality holds.

\subsection{The VAE eats the condition}
\label{sec:vae}

\input{fig_vae}

Encoding the condition with the frozen SD VAE loses most of it. An image-prompt
design such as IP-Adapter~\cite{ye2023adapter} takes a different route --- a CLIP
image encoder and a projection --- so this measurement does not speak to it. Figure~\ref{fig:vae} plots it. Passing the condition through encode--decode and
re-measuring leaves \vaeNativeOneZero{} at $r{=}0.1$, \vaeNativeThreeZero{} at
$r{=}0.3$ and \vaeNativeSevenZero{} at $r{=}0.7$. The round trip lower-bounds what the
latent retains --- a component may survive there that the frozen decoder cannot
render --- but an adapter reading that latent cannot recover what the round trip
has lost. Building the
condition at 256 and upsampling moves the structure to lower absolute frequency
and preserves far more (\vaeUpThreeZero{} at $r{=}0.3$), which is a data-pipeline
artefact worth knowing about when comparing against older setups.

\subsection{Where to inject}
\label{sec:where}

\input{fig_paths}

Table~\ref{tab:where} lists absolute configurations; each row below the control
differs from the \emph{conv / attn} row in one field. Skip-connection residuals dominate. Figure~\ref{fig:paths} decomposes them. Ablating them
away and keeping only cross-attention drops SC at $r{=}0$ from
\pathBothZero{} to \pathAttnOnlyZero{}; keeping only the skip path retains
\pathSkipOnlyZero{}. With both paths off the model collapses to
\pathAllOffZero{} SC and FID \pathFidAllOff{}, which is the control that says
the adapter is doing the work. The split is not the same on both axes: on SC the
skip path leads by roughly $2.5\times$, but on FID the two paths are almost
indistinguishable (\pathFidSkipOnly{} against \pathFidAttnOnly{}), so the
placement effect we measure is a structure-fidelity effect and not a general
image-quality one.

Capacity helps much less: widening the projection to a 128-channel stem raises
SC at $r{=}0.05$ from \scEaZeroFive{} to \scEcapZeroFive{}, and doubling again
to 256 reaches only \scEcapbZeroFive{}. A $7{\times}7$ attention window is worse
than $3{\times}3$ (\scEwinZeroFive{}). All of these vary capacity with the skip
path absent, so they bound how far capacity carries \emph{that} design rather
than settling capacity against placement in general.

Table~\ref{tab:abl} repeats the one-field ablations on the working design with
both paths on. Each is a single seed, so we read it against the main model's
three-seed spread of about 0.01 in SC. Capacity helps here too, modestly: a
256-channel stem lifts SC at $r{=}0.05$ from \scMainZeroFive{} to
\ablStemskipZeroFive{} and lowers FID at $r{=}0.1$ from \fidMainOneZero{} to
\ablFidStemskipOneZero{}, small next to adding the skip path
(\pathAttnOnlyZero{} to \pathBothZero{} at $r{=}0$). The attention window does
not help: $1{\times}1$ matches $3{\times}3$ within the seed spread or exceeds it
at every cutoff listed, and $5{\times}5$ is lower at all of them, so what the
attention sites contribute is where the condition enters, not a neighbourhood.
The zero-initialised gate is worth keeping; without it SC is lower from $r{=}0$
to $0.3$ and FID at $r{=}0.1$ is higher. The largest single change is how $r$
is drawn. A log-uniform sampler lifts SC at $r{=}0$ to \ablLogZero{} and lowers
FID there to \ablFidLogZero{}, but gives up the upper range (\ablLogFiveZero{}
at $r{=}0.5$ against \scMainFiveZero{}): it moves the dial's resolution toward
small $r$ rather than improving it everywhere. The results above use the uniform
sampler.

\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{lrrrrr}
\toprule
One field changed & \multicolumn{4}{c}{SC} & FID \\
\cmidrule(lr){2-5}
 & $r{=}0$ & $0.05$ & $0.3$ & $0.5$ & $r{=}0.1$ \\
\midrule
main model & \scMainZero & \scMainZeroFive & \scMainThreeZero & \scMainFiveZero & \fidMainOneZero \\
\quad seed sd (3 seeds) & \sSdZero & \sSdZeroFive & \sSdThreeZero & \sSdFiveZero & -- \\
\midrule
window $1{\times}1$        & \ablKoneZero & \ablKoneZeroFive & \ablKoneThreeZero & \ablKoneFiveZero & \ablFidKoneOneZero \\
window $5{\times}5$        & \ablKfiveZero & \ablKfiveZeroFive & \ablKfiveThreeZero & \ablKfiveFiveZero & \ablFidKfiveOneZero \\
no gate                    & \ablNogateZero & \ablNogateZeroFive & \ablNogateThreeZero & \ablNogateFiveZero & \ablFidNogateOneZero \\
avg-pool downsampling      & \ablDavgZero & \ablDavgZeroFive & \ablDavgThreeZero & \ablDavgFiveZero & \ablFidDavgOneZero \\
stem 64                    & \ablStemsfZero & \ablStemsfZeroFive & \ablStemsfThreeZero & \ablStemsfFiveZero & \ablFidStemsfOneZero \\
stem 256                   & \ablStemskipZero & \ablStemskipZeroFive & \ablStemskipThreeZero & \ablStemskipFiveZero & \ablFidStemskipOneZero \\
colour condition           & \ablColorZero & \ablColorZeroFive & \ablColorThreeZero & \ablColorFiveZero & \ablFidColorOneZero \\
no condition dropout       & \ablDropZero & \ablDropZeroFive & \ablDropThreeZero & \ablDropFiveZero & \ablFidDropOneZero \\
$r=u^2$                    & \ablSqrtZero & \ablSqrtZeroFive & \ablSqrtThreeZero & \ablSqrtFiveZero & \ablFidSqrtOneZero \\
$r$ log-uniform            & \ablLogZero & \ablLogZeroFive & \ablLogThreeZero & \ablLogFiveZero & \ablFidLogOneZero \\
\bottomrule
\end{tabular}
\caption{One-field ablations on the working model, skip and attention paths both on,
one training seed each; the second row is the main model's spread over three
seeds. The main model uses a $3{\times}3$ window, a zero-initialised gate,
learned $2{\times}2$ strided downsampling of the condition features at each
attention site, no stem, a grayscale condition, condition dropout 0.05, and $r$
uniform on $[0,1]$. Drawing $r=u^2$ with $u$ uniform concentrates training at
small $r$; the log-uniform sampler on $[10^{-3},1]$ does so more strongly.}
\label{tab:abl}
\end{table}


\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{llrrr}
\toprule
Cond.\ encoder & Injection & SC $r{=}0.05$ & SC $r{=}0.1$ & FID $r{=}0.1$ \\
\midrule
\multicolumn{2}{l}{\itshape no adapter (control)} & \pathAllOffZero & \pathAllOffOneZero & \pathFidAllOff \\
\midrule
frozen VAE & attn, no gate      & \scEbaseZeroFive & \scEbaseOneZero & \fidEbase \\
frozen VAE & attn               & \scEgateZeroFive & \scEgateOneZero & \fidEgate \\
conv       & attn               & \scEaZeroFive    & \scEaOneZero    & \fidEa \\
conv, 128  & attn               & \scEcapZeroFive  & \scEcapOneZero  & \fidEcap \\
conv, 256  & attn               & \scEcapbZeroFive & \scEcapbOneZero & \fidEcapb \\
conv       & attn, $7{\times}7$ & \scEwinZeroFive  & \scEwinOneZero  & \fidEwin \\
\midrule
conv       & attn + skip        & \scMainZeroFive  & \scMainOneZero  & \fidMainOneZero \\
\bottomrule
\end{tabular}
\caption{Absolute configurations, not a cumulative chain: every row below the
control differs from the \emph{conv / attn} row in exactly one field, and all
rows use a $3{\times}3$ window and a zero-initialised gate unless stated. The
main model is the last row and carries no stem, i.e.\ the minimal $1{\times}1$
projection. The first three rows sit at or below the no-adapter control, so at
that level a weak design and a run that failed to learn are not
distinguishable. Stem width with the skip path present is varied in Table~\ref{tab:abl}.}
\label{tab:where}
\end{table}

\subsection{Against per-cutoff specialists}
\label{sec:spec}

We trained four adapters at fixed $r$ and compared each against the single
randomised-$r$ model at its own cutoff. Because the single model was trained
three times, we express each gap against its three-seed spread rather than
against a within-run standard error; a gap smaller than the two-sided $t$
critical value at $df=2$ (\tailCrit{}) is not separable. The result is mixed:
the specialist wins at $r{=}0.1$ (\specOneZero{} against \sMeanOneZero{},
$\Delta{=}\specDOneZero{}$, $\specROneZero{}$ sd) and the single model wins at
$r{=}0.3$ (\sMeanThreeZero{} against \specThreeZero{}, $\specDThreeZero{}$,
$\specRThreeZero{}$ sd). Neither $r{=}0.5$ ($\specRFiveZero{}$ sd) nor
$r{=}0.7$ ($\specRSevenZero{}$ sd) separates. Our earlier single-seed reading
called $r{=}0.7$ a marginal win for the specialist; across seeds it is a tie.

One model therefore does not dominate four. What it does instead is give a dial
the specialists cannot: a model trained at $r{=}0.1$ has never seen $r{=}0.3$,
so its curve is a single point.

\subsection{Baselines}

Against pretrained ControlNet~\cite{zhang2023adding} on its own condition, over the same test images
and captions: ControlNet-Canny reaches Edge-F1 \baseCannyEdge{}, LPIPS
\baseCannyLpips{}, FID \baseCannyFid{} with \baseParams{} trainable parameters;
ControlNet-HED reaches \baseHedEdge{} / \baseHedLpips{} / \baseHedFid{}. At
$r{=}0.1$ our adapter reaches Edge-F1 \edgeMainOneZero{}, LPIPS
\lpipsMainOneZero{} and FID \fidMainOneZero{} with \ourParams{} parameters.
Two asymmetries make this a comparison of operating points rather than a
like-for-like control: the conditions differ (Canny edges against a frequency
band), and our adapter was trained on \nTrain{} FFHQ images while ControlNet is
zero-shot here and FID is measured against FFHQ statistics. On text alignment we
do not win anywhere: our best CLIP over the whole sweep is
\clipMainOneHundred{}, below ControlNet-Canny's \baseCannyClip{}
(ControlNet-HED reaches \baseHedClip{}). T2I-Adapter-Canny~\cite{mou2023adapter},
a \baseTParams{}-parameter side network and so closer to our design than a copied
encoder, reaches Edge-F1 \baseTEdge{}, LPIPS \baseTLpips{}, FID \baseTFid{} and
CLIP \baseTClip{} under the same protocol. We leave out its sketch checkpoint,
which expects a PiDiNet sketch we did not compute; fed Canny edges it would
measure that mismatch rather than the method. Training ControlNet on our condition
and data would be the matched control; we have not run it.

\subsection{Hand-drawn sketches}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/sketch}
\caption{Hand-drawn face sketches (odd columns) and what the model makes of
them (even columns), zero-shot. The drawings carry no cutoff; the model has to
read structure density off the drawing itself, which the per-image min--max
normalisation permits. Most of the scans carry the archive filename stamped
across the top, and the model redraws it --- a literal reading of the condition
it was given. Removing the whole top \bandFrac{}\% of the frame
(\bandRows{} rows), a generous over-estimate of that band, moves the score to
\bandSCcut{} (\bandDelta{}, \bandSem{} standard errors), so the band is not
what the score is measuring.}
\label{fig:sketch}
\end{figure}


Figure~\ref{fig:sketch} shows the transfer. On \sketchFaceN{} real face
drawings the model was never trained on, structure
consistency against the input drawing is \sketchFaceSC{} and CLIP score is
\sketchFaceClip{}. FID is not reported: \sketchFaceN{} images is too few for a
stable estimate, and our evaluator refuses rather than print one. We also ran
\sketchSceneN{} general-scene drawings, far outside the training domain, where
structure consistency falls to \sketchSceneSC{} and FID against FFHQ is
\sketchSceneFid{}: the model follows the drawing but does not produce a face.
That is the expected failure for a face model and we report it rather than omit it.

A drawing arrives without a cutoff, but the artist can give it one: filter the
drawing at $r$ and hand the model the result, the operation every training
condition went through. On \sdialN{} face drawings, structure consistency
against the original drawing falls from \sdialZero{} at $r{=}0$ to
\sdialOneZero{} at $r{=}0.1$, \sdialThreeZero{} at $r{=}0.3$ and
\sdialOneHundred{} at $r{=}1$, so the dial carries over to drawings; $r{=}0.5$ and
$r{=}0.7$ lie within a standard error (\sdialSemFiveZero{}) of each other.
Against the filtered drawing it was actually given, the score at $r{=}0.1$ is
\sdialFOneZero{}, well below the \scMainOneZero{} a filtered photograph yields;
we have not isolated why.

\section{Limitations}

The useful range is roughly $r \in [0, 0.3]$. Above it the condition retains
little energy --- 0.0014 of the spectrum at $r{=}0.5$ --- and structure
consistency becomes hard to interpret, since two nearly empty signals correlate
near zero whatever the model does.

Everything here is one domain (FFHQ faces) and one backbone (SD1.5), and the
three results travel differently. The dial is a property of the condition and
need not depend on the backbone, but we have not tested another. The skip-path
injection does not transfer as it stands: MMDiT backbones~\cite{esser2024scaling}
have no UNet skip connections, and control there enters through shared tokens
or LoRA~\cite{hu2021lora} modules~\cite{tan2024ominicontrol,zhang2025easycontrol,liu2025nanocontrol}.
The VAE measurement was made on the four-channel SD autoencoder only. Our adapter
is also not small by current standards: Simple-ControlNet~\cite{feng2025simplifying}
reports under 3M parameters on SD1.5, ControlNet-XS~\cite{zavadski2023controlnet}
ships much smaller control networks than ControlNet, and the DiT modules above
use a fraction of our count.

The one-field ablations of Table~\ref{tab:abl} are single-seed. The specialist comparison is honest but unflattering at $r{=}0.1$, and we have
not run a user study.

\section{Conclusion}

Randomising the bandwidth of a structure condition during training turns the
cutoff into an inference-time control, and that control is not reproducible by
scaling the adapter down. Getting there required fixing where the condition
enters the network and keeping it out of the frozen autoencoder; both were measured on the
SD1.5 autoencoder and UNet, and we expect them to carry to conditioning designs
built on that family. We have not tested others.
