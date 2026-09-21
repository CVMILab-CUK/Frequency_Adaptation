
% Placeholder for values only the authors can supply (FIX-INSTRUCTIONS 2026-09-16).
% Remove this line once `grep -c 'AUTHORINPUT{' *.inc *.tex` is 0.
\providecommand{\AUTHORINPUT}[1]{\textbf{[AUTHOR-INPUT #1]}}
\begin{abstract}
\noindent
A structure-conditioned diffusion adapter is trained for one operating point.
How closely its output follows the condition is fixed at training, and
loosening it later means another checkpoint or a lower adapter strength. We
move that choice into the condition. A single adapter on Stable Diffusion~1.5
is trained on grayscale images passed through a radial high-pass filter, with
the cutoff $r$ drawn per sample. At test time $r$ becomes a dial, much as
image-editing tools let users pick a frequency band. On FFHQ-512, raising $r$
from 0.05 to 0.3 lowers structure consistency (mean of three training seeds)
from \seedMeanZeroFive{} to \seedMeanThreeZero{} while FID stays between
\fidMainZeroFive{} and
\fidMainOneZero{}. In our samples the face layout holds while background,
clothing and colour drift from the reference. Lowering the adapter's strength
instead pays for looser structure with FID: at strength 0.50 structure is
looser than at $r{=}0.3$ under the same fixed instrument, and FID is
\fidHalfScale{} against \fidMainThreeZero{}. At $r{=}0.1$, an adapter trained
only for that cutoff still follows the structure more closely. Injecting
through the UNet skip tensors carries most of the structure, and an artist can
filter a hand drawing the same way; the output's correlation with the drawing
falls as $r$ grows.
\CCSblock
\end{abstract}

\section{Introduction}

Image editing has long given users a scale or frequency band to
turn~\cite{farbman2008edge,paris2011local}, and sketch-based face synthesis has
treated a drawing as a soft constraint rather than a
template~\cite{chen2020deepfacedrawing}, in some systems with an explicit
control over how far the sketch is trusted~\cite{yang2020deep}.
ControlNet~\cite{zhang2023adding} and its successors~\cite{mou2023adapter,zhao2023controlnet,qin2023unicontrol,li2024controlnet} moved that control into training.
An adapter trained on Canny edges follows Canny edges; one trained on depth
follows depth. Following a structure more loosely then means picking a
different checkpoint or scaling the adapter down at inference.

We change what the condition is rather than how hard the model listens to it.
Multi-scale editing removes or amplifies a band of the output
image~\cite{farbman2008edge,paris2011local}; here the band is removed from the
condition, and the generative prior decides what fills it.
The condition is the image after a radial high-pass filter with normalised
cutoff $r$. At $r{=}0$ the model receives the grayscale image and the task is
colourisation. As $r$ grows the filter removes more of the spectrum, until at
$r{=}1$ almost nothing is left: measured on \energyN{} FFHQ test images, a
cutoff of $0.05$ already discards \acRemovedZeroFive\,\% of the spectral
energy outside the constant term. Training draws $r$ per sample, so one
set of weights sees the whole range, and $r$ becomes an inference-time
parameter. At a fixed seed, raising $r$ leaves the face layout in place while
background detail thins and clothing and colour drift from the reference
(Figure~\ref{fig:dial}). Lowering the adapter's strength is a different
control: at low strength pose and framing change, and FID rises
(Sec.~\ref{sec:strength}).

Our contributions are:

\begin{itemize}\itemsep2pt
\item A frequency dial on the condition. Randomising the high-pass cutoff
      during training gives one adapter an inference-time control over
      structure adherence on $r \in [0.05, 0.3]$ at a steady FID band
      (Sec.~\ref{sec:dial}); it trails a per-cutoff specialist at $r{=}0.1$
      and leads one at $r{=}0.3$ (Sec.~\ref{sec:spec}).
\item The dial is not adapter strength. Under one fixed instrument, strength
      settings around the cutoff's structure consistency sit at higher FID
      and change pose and framing (Secs.~\ref{sec:strength}
      and~\ref{sec:fixedinst}).
\item Where the condition should enter a UNet (Sec.~\ref{sec:design}). On
      SD1.5 the skip-tensor path carries most of the structure that
      cross-attention alone does not (Secs.~\ref{sec:where}
      and~\ref{sec:skiponly}), and with attention-only injection a
      convolutional encoder reads the condition better than the frozen VAE
      (Table~\ref{tab:where}).
\end{itemize}

\section{Related work}

\paragraph{Conditional control.} Text-to-image diffusion
\cite{ho2020denoising,song2020denoising,rombach2021high,nichol2021glide,saharia2022photorealistic},
surveyed for visual computing by Po et al.~\cite{po2024state},
became spatially controllable with ControlNet~\cite{zhang2023adding}, which
reads the condition image with a small convolutional network, duplicates the
encoder and injects residuals through zero convolutions.
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
under 3M parameters, exposing a scale on that projection at inference. Each of
these fixes the condition type at training and, with it, how tightly the output
follows; an inference-time scale such as Simple-ControlNet's moves that
operating point by weakening the adapter. Our cutoff moves it through the
condition on one set of weights, and on our adapter the two controls separate
in FID (Sec.~\ref{sec:strength}).

\paragraph{Efficient control on transformer backbones.} As backbones moved from
UNets to diffusion transformers~\cite{peebles2022scalable,esser2024scaling},
control moved from duplicated encoders to lightweight modules.
ControlNeXt~\cite{peng2024controlnext} removes the parallel branch,
OminiControl~\cite{tan2024ominicontrol} reuses the backbone itself as the
condition processor, and EasyControl~\cite{zhang2025easycontrol} and
NanoControl~\cite{liu2025nanocontrol} inject conditions through
LoRA-style~\cite{hu2021lora} modules at a small fraction of the parameter cost.
Recent work also revisits how conditions enter these models:
Laytrol~\cite{huang2025laytrol} inherits the backbone's parameters to preserve
pretrained knowledge under layout control, PixelControl~\cite{lin2026pixelcontrol}
notes that the VAE's spatial compression can weaken high-frequency and low-area
condition signals, and Zhou et al.~\cite{zhou2026rethinking} replace
concatenate-and-attend with position-aligned and keyword-scoped attention in
multi-condition DiTs. On a UNet the question has a concrete form, since skip
tensors are available as an entry point: in our comparison
(Sec.~\ref{sec:where}) skip residuals carry more structure than cross-attention.

\paragraph{Frequency bands as an editing axis.} Band decompositions are a
standard editing axis in graphics: detail manipulation rescales the bands of an
edge-preserving or Laplacian decomposition~\cite{farbman2008edge,paris2011local},
and hybrid images combine the low spatial frequencies of one picture with the
high frequencies of another, so that the interpretation changes with viewing
distance~\cite{oliva2006hybrid}. In diffusion, frequency decomposition appears
mostly as an intervention on the model's own signals rather than on its
condition. FreeU~\cite{si2023freeu} rebalances skip and backbone features
spectrally at no training cost, FreSca~\cite{huang2025fresca} scales frequency
bands of the noise prediction using spatial or energy-based cutoffs,
DeCo~\cite{ma2025deco} decouples frequencies in pixel-space diffusion, and
FeRA~\cite{yin2025fera} routes adaptation by frequency energy.
FOCUS~\cite{tjio2025focus} conditions the reverse process on learned frequency
priors for test-time adaptation. Closest to us in the choice of control,
frequency bands also act as a control on image-to-image guidance:
FCDiffusion~\cite{gao2024frequency} extracts DCT bands of the source features and
switches among frequency control branches, FBSDiff~\cite{gao2024fbsdiff} sets
the guiding intensity of a reference through the bandwidth of a substituted band
without training, and ILVR~\cite{choi2021ilvr} conditions a pretrained DDPM on a
reference at a chosen downsampling factor without further learning.
Filter-guided diffusion~\cite{gu2024filter} applies a spatial filter to the
denoised estimate at inference and exposes its scale and the guidance strength
as separate controls without training; our band is instead part of what the
adapter is trained to read.
Prasain~\cite{prasain2026beyond} conditions on wavelet bands instead of edge
maps for map-to-satellite synthesis. None of them randomises the condition's
\emph{bandwidth} during training so that the band becomes a control at
inference. Randomising a conditioning parameter while training and exposing it
at inference is not itself new: SDXL~\cite{podell2023sdxl} samples crop
coordinates during data loading and surfaces them as a user control, and
Imagen~\cite{saharia2022photorealistic} draws the augmentation level of its
low-resolution condition at random in training and sweeps it at inference, there
to find the best sample quality rather than to give the user a control. Our
delta is the choice of parameter and its measured separation from adapter
strength: bandwidth and adapter strength are different controls
(Sec.~\ref{sec:strength}).

\paragraph{Drawings as soft constraints.} Sketch-based face synthesis has
treated rough drawings as soft constraints~\cite{chen2020deepfacedrawing} and,
closest to our goal, lets the user set how far a drawing is trusted through a
refinement level~\cite{yang2020deep}; diffusion-based stroke and sketch
synthesis likewise lets the user decide the level of faithfulness to the
input~\cite{cheng2023adaptively}. With pretrained diffusion models, sketch
guidance can be applied at inference through a small latent guidance predictor
instead of a dedicated encoder~\cite{voynov2023sketch}, and the noise added
before denoising balances faithfulness to the input against realism without
training~\cite{meng2022sdedit}. XDoG~\cite{winnemoller2012xdog} moves a
photograph from pencil shading to hatching and woodcut by changing the
parameters of one filter family. Our cutoff is a filter parameter of the same
kind, set on the condition and left for the generative prior to complete. Our control differs in what it varies:
the information content of the condition, set by a frequency cutoff, on one
adapter whose response we measure separately from its strength. Because the
condition is a continuum and not a fixed extractor, a single adapter can expose
it as a dial. StructDiff~\cite{he2026structdiff} targets structure-preserving,
spatially controllable single-image generation, varying samples within one
source image's own statistics while keeping its layout; we hold an external
condition instead, and the cutoff sets how much of it the output has to
reproduce.

\section{Method}

\paragraph{Condition.} Given an image $x$, we convert to grayscale, take the
2D FFT, zero every frequency inside a disc of radius $r \cdot \min(H,W)/2$,
invert, and min--max normalise to $[0,1]$. The normalisation is deliberate: it
removes the absolute scale of the filter response, so the model has to infer
structure density from spatial statistics. We hypothesise that this is what
lets a hand drawing, which carries no $r$, be used at test time.
The filter is a high-pass because the point is to keep shape and hand the rest
to the prior: edges and texture sit in the high band, tone and colour in the
low one, the split two-scale tone and detail
editing~\cite{bae2006two,farbman2008edge} works with, so a low-pass or
band-pass condition would give the model the part we want it to invent. What we
add is that the boundary is a parameter rather than a fixed extractor. The mask
is an ideal disc rather than a smooth roll-off so that $r$ names exactly which
frequencies are gone, at the cost of the ringing visible in
Figure~\ref{fig:mask}, which the model is trained on and which we did not
measure separately.
Figure~\ref{fig:mask} shows the mask, spectrum and condition at each cutoff.

\input{fig_mask}

\paragraph{Condition encoder.} A convolutional encoder reads the condition
image directly at $512^2$ and reduces it $8\times$ to the latent grid: seven
$3{\times}3$ convolutions with SiLU activations, three of them with stride 2
and followed by group normalisation, widening from 32 to 128 channels before a
final projection to 4. Each attention site reads the encoder output through a
$1{\times}1$ projection. The ablations call an optional $3{\times}3$
convolution of 64, 128 or 256 channels inserted before that projection a
\emph{stem}; the main model has none. We read the condition with this
convolutional encoder rather than the frozen VAE; Sec.~\ref{sec:vae} reports
the round-trip measurement.

\paragraph{Injection.} Encoder features enter the UNet in two places (Figure~\ref{fig:arch}). Zero-initialised
$1{\times}1$ heads produce one residual per decoder skip tensor and one for the
mid block, added through the UNet's existing ControlNet interface. In parallel,
each cross-attention site attends over a $3{\times}3$ window of the condition
features, gated by a zero-initialised per-channel vector. At $1{\times}1$ the
softmax runs over a single position, the site reduces to a gated value
projection, and structure consistency is not lower beyond the seed spread
(Sec.~\ref{sec:where}). Both paths start as
exact no-ops: with them switched off the network reproduces stock
Stable Diffusion~1.5~\cite{rombach2021high} bit for bit, which we assert before every training run.

\input{fig_arch}

\paragraph{Training.} SD1.5~\cite{rombach2021high} stays frozen. We train
\ourParams{} parameters on FFHQ-512~\cite{karras2018style} (images under their
Flickr authors' Creative Commons or public-domain licences, the dataset under
CC~BY-NC-SA~4.0) with
BLIP~\cite{li2022blip} captions, split \nTrain{} train / \nValid{} validation /
\nTest{} test by a seeded shuffle of the 70,000 images (5\% test, 2\%
validation, seed 2026). AdamW, learning rate $10^{-4}$, $\beta=(0.9,0.999)$,
cosine schedule with 500 warm-up steps, gradient clipping at 1.0, batch 32,
bf16, 12 epochs on one H200. The loss is $\epsilon$-prediction with Min-SNR
weighting~\cite{hang2023efficient}, $\gamma=5$. Text conditioning is dropped
with probability 0.1 for classifier-free guidance~\cite{ho2022classifier} and
the structure condition independently with probability
The condition is grayscale so that the filter reads one channel and a cutoff
means the same thing in every image, leaving colour to the prior;
Table~\ref{tab:abl} keeps a colour-condition row as the control for that
choice, and a no-dropout row for the structure-condition dropout of 0.05.
Training stops at 12 epochs, where the validation curve has flattened.
0.05. AdamW uses the PyTorch default weight decay of 0.01. An EMA of the
adapter (decay 0.9999) is kept with a 10-step interval; evaluation uses the
raw weights. $r$ is drawn uniformly from $[0,1]$ per sample, so one adapter
sees the whole range; we report $r \in [0.05, 0.3]$ because that is where the
control is usable, the sweep being non-monotone at the left end and not
separable across seeds at the right (Sec.~\ref{sec:dial}). No narrower training
range was tried.
Table~\ref{tab:repro} lists the
remaining settings.

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lp{0.70\columnwidth}@{}}
\toprule
Item & Setting \\
\midrule
Base model        & \texttt{stable-diffusion-v1-5/}\allowbreak\texttt{stable-diffusion-v1-5}, frozen \\
Condition encoder & seven $3{\times}3$ convolutions, SiLU, three with stride 2 and group normalisation, 32 to 128 channels, 4 output channels \\
Trainable params  & \ourParams{} (main); skip only \paramsSkipOnly{}; attention only \paramsAttnOnly{} \\
Data              & FFHQ-512, \nTrain{} / \nValid{} / \nTest{}, seeded shuffle (seed 2026); the validation split only monitors training on four images \\
Sketch data       & \texttt{a573p/}\allowbreak\texttt{Face-Sketches40}, all \sketchFaceN{}, and \texttt{rhfeiyang/}\allowbreak\texttt{photo-sketch-pair-500}; neither card states a licence \\
Optimiser         & AdamW, lr $10^{-4}$, $\beta{=}(0.9,0.999)$, weight decay 0.01, cosine, 500 warm-up steps \\
Batch / precision & 32 / bf16, gradient clipping 1.0, 12 epochs, Min-SNR $\gamma{=}5$, one H200 \\
Compute           & \trainHoursMain{} h per training run on one H200 at \trainImgSec{} images/s, peak \trainPeakGb{} GB; \sampleSecImg{} s per sampled image (DDIM 50, CFG 7.5, batch 1, 512 px), peak \samplePeakGb{} GB; \trainHoursAll{} GPU-hours over \nTrainRuns{} training runs and about \sampleHoursAll{} more for the \sampleImages{} images generated, at the measured rate, \gpuHoursAll{} in all \\
EMA               & decay 0.9999 every 10 steps; evaluation uses the raw weights \\
Dropout           & text 0.1, structure condition 0.05 \\
Training seeds    & main model 2026, 2027, 2028; specialists, ablations and skip only 2026 \\
Sampling          & DDIM, 50 steps, CFG 7.5, seed 2026; Diversity seeds $2026{+}10^5{+}i$ for image $i$ \\
Evaluation        & first 500 of the \nTest{} test images; clean-FID, clean mode, 512 px; LPIPS AlexNet against the reference; CLIP ViT-B/32 (OpenAI weights), $100\times$ cosine \\
Baselines         & \texttt{lllyasviel/}\allowbreak\texttt{sd-controlnet-canny}, \texttt{lllyasviel/}\allowbreak\texttt{sd-controlnet-hed}, \texttt{TencentARC/}\allowbreak\texttt{t2iadapter\_}\allowbreak\texttt{canny\_sd15v2}; Canny 100/200 for all; conditioning scale 1.0; default PNDM sampler, 50 steps, CFG 7.5, seed 2026 \\
Software          & Python 3.12.3, PyTorch 2.10 (NGC 25.11), diffusers 0.32.2, clean-fid 0.1.35, lpips 0.1.4, open\_clip 2.30.0 \\
\bottomrule
\end{tabular}
\caption{Settings behind every table, read from the run manifests and
evaluation scripts.}
\label{tab:repro}
\end{table}

\section{Experiments}

Tables numbered with an S prefix are in the anonymous supplementary material.

\subsection{Metrics}
\label{sec:metrics}

We evaluate on 500 of the \nTest{} held-out images with a DDIM sampler at 50
steps, CFG 7.5, seed 2026.

\paragraph{Structure consistency.} SC converts the generated image to grayscale,
applies the same high-pass filter at the same cutoff the model was
given, and takes the Pearson correlation against the given condition over the
flattened pixel vector, per image, averaged over images. Both signals are the
per-image min--max normalised filter output, so the correlation is scale free.
Two anchors bound it: feeding the ground truth back gives exactly 1.000 at every
$r$, and scoring a generation against the condition of a different test image
gives \scFloorZero{} at $r{=}0$ and \scFloorOneZero{} at $r{=}0.1$. Because the
instrument moves with $r$, a row-to-row difference mixes model behaviour with
instrument change, so we report a cutoff-independent metric beside it and a
fixed-instrument reading below.

\paragraph{Edge-F1.} Canny edges (thresholds 100 and 200 on the 8-bit grayscale
image) are taken from the generation and from the reference and matched with a
2-pixel tolerance, implemented as a $5{\times}5$ dilation of each map before
intersecting; F1 is computed per image and averaged. It does not depend on $r$,
which is why it appears in Table~\ref{tab:main}: \edgeMainZeroFive{} at
$r{=}0.05$ falling to \edgeMainOneHundred{} at $r{=}1$, so the dial survives a
cutoff-independent instrument.

\paragraph{Diversity.} Mean pairwise LPIPS (AlexNet backbone) over $K{=}4$
samples generated from one condition with different seeds; higher means the
condition constrains less. It is not independent of the LPIPS column (defined
below): across the sweep LPIPS $-$ Diversity is \divGap{}\,$\pm$\,\divGapSd{},
nearly constant while Diversity spans \divRange{}, so Diversity carries close
to the same information as LPIPS and serves as a secondary reading.

\paragraph{FID, LPIPS and CLIP.} Clean-FID~\cite{heusel2017gans,parmar2021aliased}
in its clean mode at 512 pixels, between the 500 generations and the same 500
references. The LPIPS column is AlexNet LPIPS~\cite{zhang2018unreasonable} between each generation and its
own reference, averaged over images. CLIP score is 100 times the cosine
similarity of ViT-B/32~\cite{radford2021learning} embeddings of the image and
its own caption.

\paragraph{Fixed instrument.}\label{sec:fixedinst}
Structure consistency re-extracts the condition at the same cutoff the model was
given, so the measuring instrument moves with the thing it measures. We
therefore also re-score the saved generations with the cutoff held at
$r_m{=}0.1$ throughout. This reads the saved 8-bit images, so at $r{=}0.1$ it
gives \fixInstOneZero{} where Table~\ref{tab:main}, computed before
quantisation, gives \scMainOneZero{}.

\paragraph{Code and data.} Code, configuration files, the per-run result files
and the main model's checkpoints are in an anonymous repository,
\url{https://anonymous.4open.science/w/Frequency_Adaptation-3125/}, and every
number in the paper is generated by one script from those result files.

\begin{figure*}[t]
\centering
\begin{tikzpicture}
\node[inner sep=0, anchor=south west] (d) at (0,0) {\includegraphics[width=\textwidth]{figures/dial}};
\foreach \lab [count=\i from 0] in {reference, $r{=}0$, $r{=}0.05$, $r{=}0.1$, $r{=}0.2$, $r{=}0.3$, $r{=}0.5$, $r{=}0.7$, $r{=}1$}
  \node[font=\scriptsize, anchor=south, inner sep=1.5pt]
    at ($(d.north west)!{(\i*390+192)/3504}!(d.north east)$) {\lab};
\draw[black!60, line width=0.4pt]
  ($(d.north west)!{(6*390)/3504}!(d.north east)+(0,4mm)$) -- ($(d.north east)+(0,4mm)$)
  node[font=\scriptsize, anchor=south east, inner sep=1pt] {$r > 0.3$};
\end{tikzpicture}
\caption{The main model (training seed 2026) with sampling seed
2026 and each reference's BLIP caption; along a row only $r$ changes. The rows
are the first three test images in generation order, and the tiles are the
saved generations scored in Table~\ref{tab:main}. The reference is never shown
to the model. As $r$ rises, background detail thins and clothing and colour
drift from the reference while the face layout holds; at $r{=}1$ colour
artefacts appear on the first row.
All outputs are generated by the model under study.}
\label{fig:dial}
\end{figure*}

\subsection{The dial}
\label{sec:dial}

Table~\ref{tab:main} gives the operating curve. On $r \in [0.05, 0.3]$
structure consistency falls at every step while FID stays between
\fidMainZeroFive{} and \fidMainOneZero{}, and sample diversity rises alongside.
Going from $r{=}0$ to $r{=}0.05$ structure
consistency \emph{rises} by \riseDelta{} ($z = \riseZ{}$ with the unpaired
standard error over the 500 images), so the sweep is not monotone at its left
end. Edge-F1 moves the same way, from \edgeMainZero{} to \edgeMainZeroFive{},
but we have no uncertainty for that change, so it does not separate model
behaviour from an artefact of the SC instrument.
We read the rise as the $r{=}0$ condition, the whole grayscale image, carrying
so much of the low-frequency field that it competes with the edges the adapter
must reproduce; cutting a small disc out leaves the shape easier to follow. We
did not measure this.
At the right end $r{=}0.5$ and $r{=}0.7$ are not separable across seeds
(paired $t = \tailT{}$, $df=2$, critical \tailCrit{}).

Figure~\ref{fig:dial} holds the seed and the caption fixed and varies only $r$.
Because the caption is derived from the reference, it could be carrying part of
the structure. We repeated the sweep with one generic caption, ``a photo of a
person'', for every image. Structure consistency stays close
(\genSCZero{}, \genSCOneZero{} and \genSCThreeZero{} at $r{=}0$, $0.1$ and $0.3$,
against \scMainZero{}, \scMainOneZero{} and \scMainThreeZero{} with captions)
while FID rises to \genFidZero{}--\genFidThreeZero{}, as it should once the
caption stops describing the image. The structure comes from the condition.
Figure~\ref{fig:curve} plots the curve and, below it, the cutoff and the two
scaled adapters in the structure-consistency/FID plane.

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
\multicolumn{6}{l}{\itshape $r > 0.3$} \\
0.50 & \scMainFiveZero\,$\pm$\,\semMainFiveZero   & \sSdFiveZero   & \edgeMainFiveZero  & \divMainFiveZero  & \fidMainFiveZero \\
0.70 & \scMainSevenZero\,$\pm$\,\semMainSevenZero & \sSdSevenZero  & \edgeMainSevenZero & \divMainSevenZero & \fidMainSevenZero \\
1.00 & \scMainOneHundred\,$\pm$\,\semMainOneHundred & \sSdOneHundred & \edgeMainOneHundred & \divMainOneHundred & \fidMainOneHundred \\
\bottomrule
\end{tabular}
\caption{Operating curve, 500 of the \nTest{} test images. All columns come from
training seed 2026 except seed sd, which is the standard deviation
over training seeds 2026--2028; SEM is over the 500 images. Edge-F1 is a cutoff-independent
reading of the same runs and is reported alongside SC because SC re-extracts at
the same cutoff the model was given (Sec.~\ref{sec:metrics}). FID has a single
estimate per cell and no variance, so small differences between rows are not
read as effects.}
\label{tab:main}
\end{table}

Across seeds 2026, 2027 and 2028 the curve is stable: SC at $r{=}0.05$ is
\seedMeanZeroFive{}\,$\pm$\,\seedStdZeroFive{} and at $r{=}0.3$ it is
\seedMeanThreeZero{}\,$\pm$\,\seedStdThreeZero{}.

Re-scored with the fixed instrument of Sec.~\ref{sec:fixedinst}, the curve still falls: \fixInstZeroFive{} at $r{=}0.05$, \fixInstOneZero{} at
$r{=}0.1$ and \fixInstThreeZero{} at $r{=}0.3$. Holding the instrument fixed
compresses the range, as it must, and keeps the decline.

\subsection{Adapter strength as the alternative knob}
\label{sec:strength}
An obvious alternative is to train on one cutoff and turn the adapter's
contribution down at inference, the knob lightweight control modules
expose~\cite{feng2025simplifying}. We ran it on the adapter trained at fixed
$r{=}0.1$ (Sec.~\ref{sec:spec}), scaling both injection paths, and, as a second
check, scaled the randomised-$r$ adapter at the same cutoff (Table~\ref{tab:strength};
Table~\ref{tab:strengthfull} sweeps it finely).

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
\midrule
\multicolumn{4}{l}{\itshape same generations, instrument fixed at $r_m{=}0.1$} \\
$r{=}0.05$ & \fixInstZeroFive  & \divMainZeroFive  & \fidMainZeroFive \\
$r{=}0.2$  & \fixInstTwoZero   & \divMainTwoZero   & \fidMainTwoZero \\
$r{=}0.3$  & \fixInstThreeZero & \divMainThreeZero & \fidMainThreeZero \\
\bottomrule
\end{tabular}
\caption{Strength against cutoff. Every scaled row is generated and scored at
$r{=}0.1$, so it already uses the instrument $r_m{=}0.1$; the cutoff-moved rows
are scored at their own cutoff, and the last block re-scores the same
generations with $r_m{=}0.1$ (Sec.~\ref{sec:fixedinst}). Scale 1.00 of the
fixed-$r$ adapter is a separate
inference run of the specialist of Sec.~\ref{sec:spec} (\specOneZero{} there).}
\label{tab:strength}
\end{table}

Both scaled curves leave the FFHQ image distribution as they go. At scale 0.25
the fixed-$r$ adapter reaches FID \sscFidQuarter{} and diversity
\sscDivQuarter{}, beyond anything the cutoff produces over its whole sweep,
where FID stays within \fidMainZero{}--\fidMainOneHundred{} and diversity peaks
at \divMainOneHundred{}. Turning the randomised-$r$ adapter down removes
structure while FID rises, to \fidHalfScale{} at scale 0.50 (\sscFidHalf{} for
the fixed-$r$ adapter). The fine sweep of Table~\ref{tab:strengthfull} brackets
the cutoff $r{=}0.3$, which reads \fixInstThreeZero{} at FID
\fidMainThreeZero{}, each scored at $r{=}0.1$: scale \swFVal{} gives
\swFSc{} at FID \swFFid{}, and scale \swEVal{} gives \swESc{} at FID
\swEFid{}.
We read the cost as a train--test mismatch rather than a weaker condition: the
adapter is only ever trained at scale 1.00, so turning it down puts the UNet in
a state it never saw, while $r$ was drawn per sample and moving the cutoff stays
inside the training distribution. Figure~\ref{fig:strength} agrees: at scale
0.25 both adapters still write the grayscale condition through their scaled
skip path, too weakly for the model to complete it, and the second row comes
out grey under either one. We did not measure this.
At full strength the fixed-$r{=}0.1$ adapter reaches FID \sscFidOne{} against
\fidMainOneZero{} for the randomised-$r$ adapter at the same cutoff.
Figure~\ref{fig:strength} shows what the scaled settings produce for the first
two test images.

\begin{figure*}[t]
\centering
\begin{tikzpicture}
\node[inner sep=0, anchor=south west] (s) at (0,0) {\includegraphics[width=\textwidth]{figures/strength}};
\foreach \lab [count=\i from 0] in {reference, 1.00, 0.75, 0.50, 0.25, 0.50, 0.25, $r{=}0.1$, $r{=}0.3$}
  \node[font=\scriptsize, anchor=south, inner sep=1.5pt]
    at ($(s.north west)!{(\i*256+128)/2304}!(s.north east)$) {\lab};
\foreach \a/\b/\lab in {1/5/{fixed-$r{=}0.1$ adapter, scaled}, 5/7/{randomised-$r$ at $r{=}0.1$, scaled}, 7/9/{randomised-$r$, full strength}}
  \draw[black!60, line width=0.4pt]
    ($(s.north west)!{(\a*256+12)/2304}!(s.north east)+(0,4mm)$) --
    ($(s.north west)!{(\b*256-12)/2304}!(s.north east)+(0,4mm)$)
    node[midway, font=\scriptsize, anchor=south, inner sep=1pt] {\lab};
\end{tikzpicture}
\caption{The first two test images under adapter strength and under the cutoff,
sampling seed 2026 and BLIP caption throughout. Columns 2--5: the
fixed-$r{=}0.1$ specialist at scales 1.00 to 0.25 (Table~\ref{tab:strength}).
Columns 6--7: the randomised-$r$ adapter at $r{=}0.1$, scaled to 0.50 and 0.25.
Columns 8--9: the randomised-$r$ adapter at full strength at $r{=}0.1$
(Table~\ref{tab:main}) and $r{=}0.3$, the same generations as in
Figure~\ref{fig:dial}. At scale 0.75 the fixed-$r$ adapter still follows the
reference. Lowering the strength of either adapter to 0.50 changes background
and colour, and in the second row the pose; at 0.25 the pose and framing change
in both rows and the second row turns grayscale. Moving the cutoff to $r{=}0.3$
keeps the reference layout. All outputs are generated by the model under study.}
\label{fig:strength}
\end{figure*}

\subsection{Against per-cutoff specialists}
\label{sec:spec}

We trained four adapters at fixed $r$, one run each, and compared each against
the randomised-$r$ adapter, trained three times, at its own cutoff. We divide
each gap by the randomised-$r$ adapter's three-seed standard deviation and
treat a ratio below the two-sided $t$ critical value at $df=2$ (\tailCrit{})
as not separable. Because each specialist is a single run, this threshold is
looser than a prediction interval for a new run. The result is mixed:
the specialist wins at $r{=}0.1$ (\specOneZero{} against \sMeanOneZero{},
$\Delta{=}\specDOneZero{}$, $\specROneZero{}$ sd) and the randomised-$r$ adapter wins at
$r{=}0.3$ (\sMeanThreeZero{} against \specThreeZero{}, $\specDThreeZero{}$,
$\specRThreeZero{}$ sd). Neither $r{=}0.5$ ($\specRFiveZero{}$ sd) nor
$r{=}0.7$ ($\specRSevenZero{}$ sd) separates (Table~\ref{tab:specfull}).

One model does not dominate four. A specialist trained at $r{=}0.1$ has never
seen $r{=}0.3$, so along $r$ it offers one operating point where the
randomised-$r$ adapter offers the range.

\subsection{Baselines}

Against pretrained ControlNet~\cite{zhang2023adding} and
T2I-Adapter-Canny~\cite{mou2023adapter}, over the same test images and captions,
each baseline is given Canny edges (thresholds 100 and 200) at conditioning
scale 1.0, with the pipeline's default PNDM sampler, 50 steps, CFG 7.5 and seed
2026; Table~\ref{tab:baselines} lists them next to our adapter at $r{=}0.1$,
which samples with DDIM. The ControlNet-HED checkpoint is fed the same Canny
edges rather than HED boundaries. T2I-Adapter-Canny is a side network and so
closer to our design than a copied encoder. Two asymmetries make this a
comparison of operating points rather than a like-for-like control: the
conditions differ (Canny edges against a frequency band), and our adapter was
trained on \nTrain{} FFHQ images while ControlNet is zero-shot here and FID is
measured against FFHQ statistics. On text alignment ControlNet-Canny's CLIP
\baseCannyClip{} exceeds our best over the whole sweep, \clipMainOneHundred{}.
We leave out T2I-Adapter's sketch checkpoint, which expects PiDiNet sketches as
input; fed Canny edges it would measure that mismatch rather than the method.

\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrrrrr}
\toprule
Method & Sampler & Params & Edge-F1 & LPIPS & FID & CLIP \\
\midrule
ControlNet-Canny            & PNDM & \baseParams  & \baseCannyEdge & \baseCannyLpips & \baseCannyFid & \baseCannyClip \\
ControlNet-HED, Canny input & PNDM & ---          & \baseHedEdge   & \baseHedLpips   & \baseHedFid   & \baseHedClip \\
T2I-Adapter-Canny           & PNDM & \baseTParams & \baseTEdge     & \baseTLpips     & \baseTFid     & \baseTClip \\
\midrule
ours, $r{=}0.1$             & DDIM & \ourParams   & \edgeMainOneZero & \lpipsMainOneZero & \fidMainOneZero & \clipMainOneZero \\
\bottomrule
\end{tabular}}
\caption{Pretrained baselines given Canny edges against our adapter at
$r{=}0.1$, on the same 500 test images and captions with 50 steps, CFG 7.5 and
seed 2026. Params: trainable parameters for ControlNet-Canny and ours,
side-network parameters for T2I-Adapter. The samplers differ, so the rows are
operating points, not a like-for-like comparison.}
\label{tab:baselines}
\end{table}

\subsection{Outside the training domain}
\label{sec:ood}

Everything above is FFHQ. To test whether the dial is specific to faces we ran
the sweep on the \oodN{} general-scene photographs of the public
photograph--sketch set \texttt{rhfeiyang/photo-sketch-pair-500} on Hugging Face,
whose dataset card states neither a licence nor an upstream source,
centre-cropped to $512^2$, and, as an in-domain reference under the identical
script, on the \oodN{} FFHQ test images, both with the prompt ``a high-quality
photo''. Each generation is scored in two ways: the grayscale output against
the grayscale input photograph, and the grayscale output against the filtered
condition it was given. Neither filters the output, so neither is the
structure-consistency instrument of Table~\ref{tab:main}, and the in-domain
values differ from it (Table~\ref{tab:oodfull}).

Against the input photograph the score falls from \oodOodZeroFive{} at
$r{=}0.05$ to \oodOodThreeZero{} at $r{=}0.3$ out of domain, and from
\oodIndZeroFive{} to \oodIndThreeZero{} in domain; against the filtered
condition it falls from \oodOodFZeroFive{} to \oodOodFThreeZero{} out of domain
and from \oodIndFZeroFive{} to \oodIndFThreeZero{} in domain. The dial carries
out of domain. With the adapter off, the out-of-domain images score
\oodOffOneZero{} against the input photograph at $r{=}0.1$, against
\oodOodOneZero{} with it on.

Image quality does not transfer: FID, each run measured against its own input
photographs, rises from \oodIndFidOneZero{} in domain to \oodOodFidOneZero{}
out of it at $r{=}0.1$. Against the filtered condition, the same two runs score
\oodIndFOneZero{} and \oodOodFOneZero{}, and they stay this close at every
cutoff. Both are far above chance: scored against a different input's filtered
condition the same generations reach \oodFloorIndOneZero{} and
\oodFloorOodOneZero{} at $r{=}0.1$, and no reading in either run is fewer than
\oodFloorMinSem{} standard errors above its own floor. Condition following and
image quality therefore come apart: on the range we claim, the dial holds
structure about as closely outside the training domain as inside it, while FID
does not transfer.

\subsection{Hand-drawn sketches}
\begin{figure}[t]
\centering
\setlength{\tabcolsep}{0pt}%
\begin{tabular}{@{}r@{\hspace{2pt}}c@{\hspace{1pt}}c@{\hspace{1pt}}c@{\hspace{1pt}}c@{}}
\rotatebox{90}{\scriptsize\hspace{1mm}drawing (input)} &
\includegraphics[width=0.235\columnwidth,trim={0 0 1792 0},clip]{figures/sketch} &
\includegraphics[width=0.235\columnwidth,trim={512 0 1280 0},clip]{figures/sketch} &
\includegraphics[width=0.235\columnwidth,trim={1024 0 768 0},clip]{figures/sketch} &
\includegraphics[width=0.235\columnwidth,trim={1536 0 256 0},clip]{figures/sketch} \\[1pt]
\rotatebox{90}{\scriptsize\hspace{1mm}output, unedited} &
\includegraphics[width=0.235\columnwidth,trim={256 0 1536 0},clip]{figures/sketch} &
\includegraphics[width=0.235\columnwidth,trim={768 0 1024 0},clip]{figures/sketch} &
\includegraphics[width=0.235\columnwidth,trim={1280 0 512 0},clip]{figures/sketch} &
\includegraphics[width=0.235\columnwidth,trim={1792 0 0 0},clip]{figures/sketch}
\end{tabular}
\caption{The first four of the \sketchFaceN{} face drawings in file order (top
row), and the unedited zero-shot outputs of the main
model for them (bottom row; prompt ``a high-quality photo of a face''). The
drawings carry no cutoff; the model has to
read structure density off the drawing itself, which the per-image min--max
normalisation permits. Most of the scans carry the archive filename stamped
across the top, and the model redraws it --- a literal reading of the condition
it was given. Removing the whole top \bandFrac{}\% of the frame
(\bandRows{} rows), a generous over-estimate of that band, moves the score to
\bandSCcut{} (\bandDelta{}, \bandSem{} standard errors), so the band is not
what the score is measuring.}
\label{fig:sketch}
\end{figure}


% <<AUTHOR-INPUT A28: 이 40장이 CUFS인지 CUFSF인지 확정이 필요하다. 스캔에 찍힌 파일명이 40장 모두
% 00001fb010_930831.jpg 형식(피험자번호 + fa/fb 정면 코드 + 촬영일 930831)으로 FERET 규약이며,
% FERET 사진으로 만든 CUHK 스케치 DB는 CUFSF다. CUFSF라면 (a) 본문의 database 이름을 고치고
% (b) CUFSF 페이지가 요구하는 Zhang, Wang & Tang, CVPR 2011을 AUTHOR-CITE 절차로 추가하며
% (c) 게재 조건은 FERET을 따른다. 게재 불가로 판단되면 Fig.6 타일을 교체해야 한다>>
Figure~\ref{fig:sketch} shows the transfer. On all \sketchFaceN{} real face
drawings of the public Hugging Face set
\texttt{a573p/}\allowbreak\texttt{Face-Sketches40}, which its card describes as
images of the CUHK Face Sketch database and for which it states no licence.
Every scan is stamped with a FERET filename, so they are from the FERET-based
CUFSF set~\cite{wang2009face,zhang2011coupled} rather than the photographic
subsets of CUFS. The drawings, which the
model was never trained on, prompted with ``a high-quality photo of a face'', the
grayscale output correlates with the input drawing at \sketchFaceSC{} and CLIP score is
\sketchFaceClip{}. FID is not reported: \sketchFaceN{} images is too few for a
stable estimate, and our evaluator refuses rather than print one. We also ran
\sketchSceneN{} general-scene drawings, the sketch half of the public
photograph--sketch set \texttt{rhfeiyang/photo-sketch-pair-500} on Hugging Face,
far outside the training domain and with the same face prompt, where
that correlation falls to \sketchSceneSC{} and FID against FFHQ is
\sketchSceneFid{}: the model follows the drawing but does not produce a face.
It was trained on nothing else.

Sketch-based face editing has divided a drawing's role before, taking geometry
from the sketch and appearance from a reference~\cite{chen2021deepfaceediting},
or splitting it into separately applicable sketch
concepts~\cite{xiao2024customsketching}. The cutoff divides it by frequency and
needs no reference image.

A drawing arrives without a cutoff, but the artist can give it one: filter the
drawing at $r$ and hand the model the result, the operation every training
condition went through. On \sdialN{} face drawings, the grayscale output's
correlation with the original drawing falls from \sdialZero{} at $r{=}0$ to
\sdialOneZero{} at $r{=}0.1$, \sdialThreeZero{} at $r{=}0.3$ and
\sdialOneHundred{} at $r{=}1$; $r{=}0.5$ and $r{=}0.7$ lie within a standard
error (\sdialSemFiveZero{}) of each other.
As intended use rather than measurement: $r{=}0.05$ is for holding the face as
closely as the adapter can hold it, $r{=}0.3$ for keeping the layout while the
detail is redrawn, and above that the layout itself is released. Changing $r$
costs one regeneration, \sampleSecImg{} s per image on one H200
(Table~\ref{tab:repro}).

Against the filtered drawing the model was actually given, the same generations
score \sdialFZeroFive{} at $r{=}0.05$, \sdialFOneZero{} at $r{=}0.1$ and
\sdialFThreeZero{} at $r{=}0.3$. Scored by the same script, \oodN{} FFHQ test
photographs prompted with ``a high-quality photo'' reach \oodIndFZeroFive{},
\oodIndFOneZero{} and \oodIndFThreeZero{} (Sec.~\ref{sec:ood}), so a filtered
drawing is followed less closely than a filtered photograph at every cutoff; at
$r{=}0.3$ the drawing reaches less than half the photograph's score
(\sdialFThreeZero{} against \oodIndFThreeZero{}), though the prompts and set
sizes differ (Table~\ref{tab:sketchspec}).

\section{What the design needs}
\label{sec:design}

\subsection{How much of the condition survives the frozen VAE}
\label{sec:vae}

\input{fig_vae}

Figure~\ref{fig:vae} measures how much of a high-pass condition survives the
frozen SD VAE. After an encode--decode round trip, the re-extracted condition
correlates with the original at \vaeNativeOneZero{} for $r{=}0.1$,
\vaeNativeThreeZero{} for $r{=}0.3$ and \vaeNativeSevenZero{} for $r{=}0.7$.
The round trip lower-bounds what the latent retains: a component may survive in
the latent that the frozen decoder cannot render. The frozen-VAE designs in
Table~\ref{tab:where} inject through attention only and reach structure
consistency \scEgateOneZero{} at $r{=}0.1$, where the round trip still keeps a
correlation of \vaeNativeOneZero{}, so condition loss in the round trip does not
by itself account for their result. An image-prompt design such as
IP-Adapter~\cite{ye2023adapter} reads a CLIP image embedding instead, so this
measurement does not speak to it. Building the condition at 256 and upsampling
moves the structure to lower absolute frequency, and the round trip then keeps
\vaeUpThreeZero{} at $r{=}0.3$ against \vaeNativeThreeZero{} at native
resolution, a data-pipeline artefact worth knowing about when comparing against
older setups.

\subsection{Where to inject}
\label{sec:where}

\input{fig_paths}

Table~\ref{tab:where} lists absolute configurations. Each row below the control
changes one field of the \emph{conv / attn} row, except the frozen-VAE row
without a zero-initialised gate, which changes two. Figure~\ref{fig:paths}
decomposes the main model by switching paths off at test time. Keeping only
cross-attention drops SC at $r{=}0$ from \pathBothZero{} to \pathAttnOnlyZero{};
keeping only the skip path retains \pathSkipOnlyZero{}. With both paths off the
model falls to \pathAllOffZero{} SC and FID \pathFidAllOff{}, the control that
says the adapter is doing the work. The two single paths are close on FID
(\pathFidSkipOnly{} against \pathFidAttnOnly{}) and far apart on SC. How the
two paths combine also changes with the cutoff: at $r{=}0$ their gains over the
control roughly add up, while at $r{=}0.3$ the two together exceed that sum. We
read this as the paths carrying the same information when the condition is
dense and different information when it is sparse, the skip residuals placing
local detail and cross-attention saying where it goes. We did not measure this.

Within the attention-only design, a 128-channel stem raises SC at $r{=}0.05$
from \scEaZeroFive{} to \scEcapZeroFive{}, and a 256-channel stem reaches
\scEcapbZeroFive{} with \paramsCapb{} trainable parameters against
\paramsAttnOnly{} without a stem; the skip path trained alone, with
\paramsSkipOnly{} parameters, reaches \skipOnlyZeroFive{}
(Sec.~\ref{sec:skiponly}). A $7{\times}7$ window gives \scEwinZeroFive{}, far below
the $3{\times}3$ design (\scEaZeroFive{}) and close to the no-adapter control
(\pathAllOffZeroFive{}). All of these vary capacity with the skip path absent,
so they measure capacity within the attention-only design.

Table~\ref{tab:abl} repeats the one-field ablations on the main model with both
paths on. Each is a single seed, so we read it directionally against the main
model's three-seed spread at each cutoff (\sSdZero{} at $r{=}0$, \sSdZeroFive{}
at $r{=}0.05$). Capacity helps here too. At $r{=}0$ a 256-channel stem lifts SC
from \scMainZero{} to \ablStemskipZero{}, several times the seed spread there;
adding the skip path to attention alone moves it from \pathAttnOnlyZero{} to
\pathBothZero{}. At $r{=}0.05$ the stem gain shrinks (\scMainZeroFive{} to
\ablStemskipZeroFive{}), and FID at $r{=}0.1$ drops from \fidMainOneZero{} to
\ablFidStemskipOneZero{}. The attention window does
not help: $1{\times}1$ matches $3{\times}3$ within the seed spread or exceeds it
at every cutoff listed, and $5{\times}5$ is lower at all of them, so what the
attention sites contribute is where the condition enters, not a neighbourhood.
The zero-initialised gate is worth keeping; without it SC is lower from $r{=}0$
to $0.3$ and FID at $r{=}0.1$ is higher. The largest single change is how $r$
is drawn. A log-uniform sampler lifts SC at $r{=}0$ to \ablLogZero{} and lowers
FID there to \ablFidLogZero{}, but gives up the upper range (\ablLogFiveZero{}
at $r{=}0.5$ against \scMainFiveZero{}): it moves the dial's resolution toward
small $r$ rather than improving it everywhere. The results above use the uniform
sampler. The main model keeps the $3{\times}3$ window and no stem, although the
$1{\times}1$ and stem variants score higher at several cutoffs in
Table~\ref{tab:abl}, because these one-field ablations were trained after
the main configuration and its three seeds had been fixed. The same holds for
the $r{=}u^2$ sampler. It concentrates training at small $r$ and scores higher
structure consistency than the main model at each measured cutoff of the
claimed range (Tables~\ref{tab:abl} and~\ref{tab:ablfull}). Its FID is lower at
$r{=}0.1$ and within single-estimate noise at $r{=}0.05$ and $0.3$
(Table~\ref{tab:ablfid}). We leave it as an ablation rather than adopt it: the
sampler is one of the fields this section varies, and choosing it on these
numbers would be choosing it on the test set.

\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{2pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrrr}
\toprule
Cond.\ encoder & Injection & SC $r{=}0.05$ & SC $r{=}0.1$ & FID $r{=}0.1$ \\
\midrule
\multicolumn{2}{l}{\itshape no adapter (control)} & \pathAllOffZeroFive & \pathAllOffOneZero & \pathFidAllOffOneZero \\
\midrule
frozen VAE & attn, no zero-init gate & \scEbaseZeroFive & \scEbaseOneZero & \fidEbase \\
frozen VAE & attn               & \scEgateZeroFive & \scEgateOneZero & \fidEgate \\
conv       & attn               & \scEaZeroFive    & \scEaOneZero    & \fidEa \\
conv + 128-ch stem & attn               & \scEcapZeroFive  & \scEcapOneZero  & \fidEcap \\
conv + 256-ch stem & attn               & \scEcapbZeroFive & \scEcapbOneZero & \fidEcapb \\
conv       & attn, $7{\times}7$ & \scEwinZeroFive  & \scEwinOneZero  & \fidEwin \\
\midrule
conv       & attn + skip        & \scMainZeroFive  & \scMainOneZero  & \fidMainOneZero \\
\bottomrule
\end{tabular}}
\caption{Absolute configurations, not a cumulative chain. Rows below the
control change one field of the \emph{conv / attn} row; the first frozen-VAE
row changes two (encoder, and no zero-initialised gate). All rows use a
$3{\times}3$ window and a zero-initialised gate unless stated. The
main model is the last row and carries no stem; its attention sites read the
encoder output through a $1{\times}1$ projection. The frozen-VAE rows use
attention injection only. They and the $7{\times}7$ row stay close to the
no-adapter control at both cutoffs, so at that level a weak design and a run
that failed to learn are not distinguishable. Stem width with the skip path present is varied in Table~\ref{tab:abl}.}
\label{tab:where}
\end{table}

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
$1{\times}1$ window        & \ablKoneZero & \ablKoneZeroFive & \ablKoneThreeZero & \ablKoneFiveZero & \ablFidKoneOneZero \\
$5{\times}5$ window        & \ablKfiveZero & \ablKfiveZeroFive & \ablKfiveThreeZero & \ablKfiveFiveZero & \ablFidKfiveOneZero \\
no zero-init gate          & \ablNogateZero & \ablNogateZeroFive & \ablNogateThreeZero & \ablNogateFiveZero & \ablFidNogateOneZero \\
average-pool downsampling  & \ablDavgZero & \ablDavgZeroFive & \ablDavgThreeZero & \ablDavgFiveZero & \ablFidDavgOneZero \\
64-ch stem                 & \ablStemsfZero & \ablStemsfZeroFive & \ablStemsfThreeZero & \ablStemsfFiveZero & \ablFidStemsfOneZero \\
256-ch stem                & \ablStemskipZero & \ablStemskipZeroFive & \ablStemskipThreeZero & \ablStemskipFiveZero & \ablFidStemskipOneZero \\
colour condition           & \ablColorZero & \ablColorZeroFive & \ablColorThreeZero & \ablColorFiveZero & \ablFidColorOneZero \\
no condition dropout       & \ablDropZero & \ablDropZeroFive & \ablDropThreeZero & \ablDropFiveZero & \ablFidDropOneZero \\
$r{=}u^2$ sampler          & \ablSqrtZero & \ablSqrtZeroFive & \ablSqrtThreeZero & \ablSqrtFiveZero & \ablFidSqrtOneZero \\
log-uniform sampler        & \ablLogZero & \ablLogZeroFive & \ablLogThreeZero & \ablLogFiveZero & \ablFidLogOneZero \\
\bottomrule
\end{tabular}
\caption{One-field ablations on the main model, skip and attention paths both on,
one training seed each; the second row is the main model's spread over three
seeds. The main model uses a $3{\times}3$ window, a zero-initialised gate,
learned $2{\times}2$ strided downsampling of the condition features at each
attention site, no stem before the $1{\times}1$ projection, a grayscale condition, condition dropout 0.05, and $r$
uniform on $[0,1]$. Drawing $r=u^2$ with $u$ uniform concentrates training at
small $r$; the log-uniform sampler on $[10^{-3},1]$ does so more strongly.}
\label{tab:abl}
\end{table}



\subsection{The skip path alone}
\label{sec:skiponly}

Section~\ref{sec:where} ablates the two injection paths on a model trained with
both. That confounds placement with capacity: the cross-attention branch is
still present and trained, just switched off at test time. We therefore trained
the skip path from scratch, with the attention branch removed entirely.

It reaches SC \skipOnlyOneZero{} at $r{=}0.1$ against \scMainOneZero{} for both
paths and \pathAttnOnlyOneZero{} for cross-attention alone, at FID
\skipOnlyFidOneZero{} against \fidMainOneZero{}. Trained on its own, the skip
path recovers most but not all of the main model, so the attention branch
adds something the skip path does not supply. The skip-only model trains
\paramsSkipOnly{} parameters against \ourParams{} for both paths, and the
attention-only design with a 256-channel stem, at \paramsCapb{} parameters,
reaches \scEcapbOneZero{} at $r{=}0.1$ (Table~\ref{tab:where}). The full curve
is in Table~\ref{tab:skiponly}.

\begin{table}[t]\centering\footnotesize
\begin{tabular}{lrrr}
\toprule
$r$ & skip only & both paths & attention only \\
\midrule
0.05 & \skipOnlyZeroFive  & \scMainZeroFive  & --- \\
0.10 & \skipOnlyOneZero   & \scMainOneZero   & \pathAttnOnlyOneZero \\
0.20 & \skipOnlyTwoZero   & \scMainTwoZero   & --- \\
0.30 & \skipOnlyThreeZero & \scMainThreeZero & \pathAttnOnlyThreeZero \\
0.50 & \skipOnlyFiveZero  & \scMainFiveZero  & --- \\
0.70 & \skipOnlySevenZero & \scMainSevenZero & --- \\
\bottomrule
\end{tabular}
\caption{Structure consistency for a model trained with the skip path only,
against the main model and against its cross-attention branch measured in
isolation. The last column comes from switching a path off at test time, so it
is not a trained-from-scratch comparison.}
\label{tab:skiponly}
\end{table}

\section{Limitations}

Everything here is one domain (FFHQ faces) and one backbone (SD1.5), and the
three results depend on that backbone to different degrees. The dial is defined
on the condition alone; we measured it on SD1.5 with FFHQ-512.
SDXL~\cite{podell2023sdxl} keeps the UNet skip tensors and is the closest next
test. The skip-path injection is specific to UNets: MMDiT
backbones~\cite{esser2024scaling} have no UNet skip connections, and control
there enters through shared tokens or LoRA~\cite{hu2021lora}
modules~\cite{tan2024ominicontrol,zhang2025easycontrol,liu2025nanocontrol}. The
VAE measurement covers the four-channel SD autoencoder.

We claim the dial on $r \in [0.05, 0.3]$. Above that range the condition retains
little energy (\acRetainedFiveZero{} of the non-constant spectral energy at
$r{=}0.5$), and
structure consistency becomes hard to interpret, since two nearly empty signals
correlate near zero whatever the model does. FID is a single estimate per cell
with no variance, so we leave gaps inside the band of the cutoff sweep
(\fidMainZero{}--\fidMainOneHundred{}) uninterpreted. The strength setting's FID
is measured against FFHQ statistics; at scales of 0.50 and below the samples of
Figure~\ref{fig:strength} depart from the reference layout rather than visibly
degrading, so the rise
reflects leaving that distribution at least in part, and we report no
no-reference quality measure. The one-field ablations of
Table~\ref{tab:abl} are single-seed, and the per-image normalisation of the
condition was fixed by design and never ablated. At $r{=}0.1$ the fixed-cutoff
specialist leads the randomised-$r$ adapter by $\specROneZero{}$ seed sd
(Sec.~\ref{sec:spec}). The frozen VAE was never paired with the skip path, so in
Table~\ref{tab:where} the VAE result is confounded with the weakness of
attention-only injection. The sketch results have no adapter-off control. The
ControlNet comparison is between operating points; a ControlNet trained on our
condition and data would be the like-for-like control, and we did not train
one. We do not compare against a sketch-reliability control~\cite{yang2020deep}
or against noise-level editing~\cite{meng2022sdedit}, training-free filter
guidance~\cite{gu2024filter} or geometry--appearance face
editing~\cite{chen2021deepfaceediting}; they act on different inputs and would
need their own protocol.

The adapter trains \ourParams{} parameters, fewer than ControlNet and more than
recent lightweight control modules: Simple-ControlNet~\cite{feng2025simplifying}
reports under 3M parameters on SD1.5, ControlNet-XS~\cite{zavadski2023controlnet}
ships much smaller control networks than ControlNet, and the DiT modules above
use a fraction of our count. We make no efficiency claim. We evaluate the dial
with image metrics only, and neither SC nor Edge-F1 separates layout from
appearance, so the observation that layout holds while appearance changes
rests on the samples in Figures~\ref{fig:dial} and~\ref{fig:strength}. Whether
equal steps in $r$ are perceived as equal changes, and whether artists find
the control useful, is not measured here.

\section{Conclusion}

Randomising the bandwidth of a structure condition during training turns the
cutoff into an inference-time control, and that control is not reproducible by
scaling this adapter's strength. One checkpoint covers $r \in [0.05, 0.3]$ at
FID between \fidMainZeroFive{} and \fidMainOneZero{}; across that range our
samples keep the face layout while background, clothing and colour follow the
reference less. Lowering strength to a comparable looseness costs FID
(Sec.~\ref{sec:strength}). The checkpoint trails a specialist at $r{=}0.1$ and
leads one at $r{=}0.3$ (Sec.~\ref{sec:spec}). The design reads the condition
with a convolutional encoder rather than the frozen autoencoder and injects it
through the UNet skip tensors and cross-attention, both choices measured on
SD1.5.

\ifx\AppendixAfterReferences\undefined\input{fa_appendix.inc}\fi
