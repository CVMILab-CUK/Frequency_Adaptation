\appendix
\section{Full per-cutoff results}
\suppressfloats[t]

The main text reports each experiment at the cutoff that carries its argument.
The tables here give every cutoff that was measured, from the same
\texttt{results.json} files the rest of the paper is generated from.
Tables~\ref{tab:ablfull} and~\ref{tab:ablfid} extend Table~\ref{tab:abl};
Table~\ref{tab:allmetrics} extends Table~\ref{tab:main};
Table~\ref{tab:skipseed} extends Sec.~\ref{sec:skiponly} and the seed spread;
Table~\ref{tab:strengthfull} extends Sec.~\ref{sec:strength};
Table~\ref{tab:specfull} extends Sec.~\ref{sec:spec};
Tables~\ref{tab:oodfull} and~\ref{tab:sketchspec} extend Sec.~\ref{sec:ood} and the hand-drawn sketches;
Table~\ref{tab:repro} lists the settings behind all of them.

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lrrrrr}
\toprule
variant & $r{=}0$ & 0.05 & 0.1 & 0.3 & 0.5 \\
\midrule
main model        & \scMainZero      & \scMainZeroFive      & \scMainOneZero      & \scMainThreeZero      & \scMainFiveZero \\
$1{\times}1$ window  & \ablKoneZero     & \ablKoneZeroFive     & \ablKoneOneZero     & \ablKoneThreeZero     & \ablKoneFiveZero \\
$5{\times}5$ window  & \ablKfiveZero    & \ablKfiveZeroFive    & \ablKfiveOneZero    & \ablKfiveThreeZero    & \ablKfiveFiveZero \\
64-ch stem           & \ablStemsfZero   & \ablStemsfZeroFive   & \ablStemsfOneZero   & \ablStemsfThreeZero   & \ablStemsfFiveZero \\
256-ch stem          & \ablStemskipZero & \ablStemskipZeroFive & \ablStemskipOneZero & \ablStemskipThreeZero & \ablStemskipFiveZero \\
average-pool downsampling & \ablDavgZero     & \ablDavgZeroFive     & \ablDavgOneZero     & \ablDavgThreeZero     & \ablDavgFiveZero \\
colour condition     & \ablColorZero    & \ablColorZeroFive    & \ablColorOneZero    & \ablColorThreeZero    & \ablColorFiveZero \\
log-uniform sampler   & \ablLogZero      & \ablLogZeroFive      & \ablLogOneZero      & \ablLogThreeZero      & \ablLogFiveZero \\
$r{=}u^2$ sampler  & \ablSqrtZero     & \ablSqrtZeroFive     & \ablSqrtOneZero     & \ablSqrtThreeZero     & \ablSqrtFiveZero \\
no condition dropout & \ablDropZero     & \ablDropZeroFive     & \ablDropOneZero     & \ablDropThreeZero     & \ablDropFiveZero \\
no zero-init gate    & \ablNogateZero   & \ablNogateZeroFive   & \ablNogateOneZero   & \ablNogateThreeZero   & \ablNogateFiveZero \\
\midrule
\multicolumn{6}{l}{\itshape designs not adopted (Sec.~\ref{sec:vae})} \\
256px source pipeline & --- & \scEresZeroFive & \scEresOneZero & \scEresThreeZero & --- \\
VAE condition         & --- & \scEbaseZeroFive & \scEbaseOneZero & \scEbaseThreeZero & --- \\
\bottomrule
\end{tabular}
% <<AUTHOR-INPUT A21: "256px source pipeline" 실행(\scEres*)의 주입 구성 -- attention only인지, zero-init gate 유무, skip 경로 유무>>
\caption{Structure consistency for every one-field ablation, at every measured
cutoff. Each row of the first block below the main model changes one field of
the main model. The last block holds two designs the paper argues against:
reading the condition through the frozen VAE, which also uses attention-only
injection without a zero-initialised gate (the first frozen-VAE row of
Table~\ref{tab:where}), and building the condition at 256 and upsampling (FID
\fidEres{} at $r{=}0.1$; injection \AUTHORINPUT{A21}).}
\label{tab:ablfull}
\end{table}

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrrr}
\toprule
variant & $r{=}0$ & 0.05 & 0.1 & 0.3 & 0.5 \\
\midrule
main model          & \fidMainZero & \fidMainZeroFive & \fidMainOneZero & \fidMainThreeZero & \fidMainFiveZero \\
$1{\times}1$ window    & \ablFidKoneZero & \ablFidKoneZeroFive & \ablFidKoneOneZero & \ablFidKoneThreeZero & \ablFidKoneFiveZero \\
$5{\times}5$ window    & \ablFidKfiveZero & \ablFidKfiveZeroFive & \ablFidKfiveOneZero & \ablFidKfiveThreeZero & \ablFidKfiveFiveZero \\
64-ch stem             & \ablFidStemsfZero & \ablFidStemsfZeroFive & \ablFidStemsfOneZero & \ablFidStemsfThreeZero & \ablFidStemsfFiveZero \\
256-ch stem            & \ablFidStemskipZero & \ablFidStemskipZeroFive & \ablFidStemskipOneZero & \ablFidStemskipThreeZero & \ablFidStemskipFiveZero \\
average-pool downsampling   & \ablFidDavgZero & \ablFidDavgZeroFive & \ablFidDavgOneZero & \ablFidDavgThreeZero & \ablFidDavgFiveZero \\
colour condition       & \ablFidColorZero & \ablFidColorZeroFive & \ablFidColorOneZero & \ablFidColorThreeZero & \ablFidColorFiveZero \\
log-uniform sampler     & \ablFidLogZero & \ablFidLogZeroFive & \ablFidLogOneZero & \ablFidLogThreeZero & \ablFidLogFiveZero \\
$r{=}u^2$ sampler    & \ablFidSqrtZero & \ablFidSqrtZeroFive & \ablFidSqrtOneZero & \ablFidSqrtThreeZero & \ablFidSqrtFiveZero \\
no condition dropout   & \ablFidDropZero & \ablFidDropZeroFive & \ablFidDropOneZero & \ablFidDropThreeZero & \ablFidDropFiveZero \\
no zero-init gate      & \ablFidNogateZero & \ablFidNogateZeroFive & \ablFidNogateOneZero & \ablFidNogateThreeZero & \ablFidNogateFiveZero \\
\bottomrule
\end{tabular}
\caption{FID for the same one-field ablations as Table~\ref{tab:ablfull}, one
training seed each. FID is a single estimate per cell with no variance
(Table~\ref{tab:main}), so small differences are not read as effects; the largest
departures are the log-uniform and $r{=}u^2$ samplers at $r{=}0$
(\ablFidLogZero{} and \ablFidSqrtZero{} against \fidMainZero{}).}
\label{tab:ablfid}
\end{table}

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
$r$ & SC & FID & LPIPS & CLIP & Edge-F1 & Div. \\
\midrule
0.00 & \scMainZero       & \fidMainZero       & \lpipsMainZero       & \clipMainZero       & \edgeMainZero       & \divMainZero \\
0.05 & \scMainZeroFive   & \fidMainZeroFive   & \lpipsMainZeroFive   & \clipMainZeroFive   & \edgeMainZeroFive   & \divMainZeroFive \\
0.10 & \scMainOneZero    & \fidMainOneZero    & \lpipsMainOneZero    & \clipMainOneZero    & \edgeMainOneZero    & \divMainOneZero \\
0.20 & \scMainTwoZero    & \fidMainTwoZero    & \lpipsMainTwoZero    & \clipMainTwoZero    & \edgeMainTwoZero    & \divMainTwoZero \\
0.30 & \scMainThreeZero  & \fidMainThreeZero  & \lpipsMainThreeZero  & \clipMainThreeZero  & \edgeMainThreeZero  & \divMainThreeZero \\
0.50 & \scMainFiveZero   & \fidMainFiveZero   & \lpipsMainFiveZero   & \clipMainFiveZero   & \edgeMainFiveZero   & \divMainFiveZero \\
0.70 & \scMainSevenZero  & \fidMainSevenZero  & \lpipsMainSevenZero  & \clipMainSevenZero  & \edgeMainSevenZero  & \divMainSevenZero \\
1.00 & \scMainOneHundred & \fidMainOneHundred & \lpipsMainOneHundred & \clipMainOneHundred & \edgeMainOneHundred & \divMainOneHundred \\
\bottomrule
\end{tabular}
\caption{Every metric for the main model, at every cutoff. The SC and FID
columns are the ones plotted in Figure~\ref{fig:curve}.}
\label{tab:allmetrics}
\end{table}

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
$r$ & skip only & skip only FID & seed mean & seed s.d. \\
\midrule
0.00 & \skipOnlyZero      & \skipOnlyFidZero      & \sMeanZero        & \sSdZero \\
0.05 & \skipOnlyZeroFive  & \skipOnlyFidZeroFive  & \seedMeanZeroFive & \seedStdZeroFive \\
0.10 & \skipOnlyOneZero   & \skipOnlyFidOneZero   & \seedMeanOneZero  & \seedStdOneZero \\
0.20 & \skipOnlyTwoZero   & \skipOnlyFidTwoZero   & \seedMeanTwoZero  & \seedStdTwoZero \\
0.30 & \skipOnlyThreeZero & \skipOnlyFidThreeZero & \seedMeanThreeZero& \seedStdThreeZero \\
0.50 & \skipOnlyFiveZero  & \skipOnlyFidFiveZero  & \seedMeanFiveZero & \seedStdFiveZero \\
0.70 & \skipOnlySevenZero & \skipOnlyFidSevenZero & \seedMeanSevenZero& \seedStdSevenZero \\
1.00 & \skipOnlyOneHundred& \skipOnlyFidOneHundred& \sMeanOneHundred  & \sSdOneHundred \\
\bottomrule
\end{tabular}
\caption{Left: the skip path trained alone (Sec.~\ref{sec:skiponly}). Right:
mean and standard deviation of structure consistency over seeds 2026, 2027 and
2028 for the main model.}
\label{tab:skipseed}
\end{table}

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}
\toprule
adapter strength & SC & FID & SC, seed 2027 \\
\midrule
\swAVal & \swASc & \swAFid & --- \\
\swBVal & \swBSc & \swBFid & --- \\
\swCVal & \swCSc & \swCFid & \swCScB \\
\swDVal & \swDSc & \swDFid & --- \\
\swEVal & \swESc & \swEFid & \swEScB \\
\swFVal & \swFSc & \swFFid & --- \\
\swGVal & \swGSc & \swGFid & \swGScB \\
1.00    & \scMainOneZero & \fidMainOneZero & --- \\
\bottomrule
\end{tabular}
\caption{The strength knob on the randomised-$r$ adapter at $r{=}0.1$, swept
finely. Each scaled point below 1.00 pays for lower structure consistency with
FID; over $r \in [0.05, 0.3]$ the cutoff dial keeps FID between
\fidMainZeroFive{} and \fidMainOneZero{} (Table~\ref{tab:allmetrics}). Scales
\swCVal{}, \swEVal{} and \swGVal{} were repeated with sampling seed 2027 on the
same checkpoint.}
\label{tab:strengthfull}
\end{table}

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrr}
\toprule
 & & \multicolumn{2}{c}{one model, 3 seeds} & & \\
\cmidrule(lr){3-4}
$r$ & specialist & mean & sd & difference & diff.\,/\,sd \\
\midrule
0.10 & \specOneZero   & \sMeanOneZero   & \sSdOneZero   & \specDOneZero   & \specROneZero \\
0.30 & \specThreeZero & \sMeanThreeZero & \sSdThreeZero & \specDThreeZero & \specRThreeZero \\
0.50 & \specFiveZero  & \sMeanFiveZero  & \sSdFiveZero  & \specDFiveZero  & \specRFiveZero \\
0.70 & \specSevenZero & \sMeanSevenZero & \sSdSevenZero & \specDSevenZero & \specRSevenZero \\
\bottomrule
\end{tabular}
\caption{Per-cutoff specialists (one training run each, seed 2026) against the
single randomised-$r$ model averaged over training seeds 2026--2028, each at its
own cutoff. Difference is specialist minus the three-seed mean; the last column
divides it by the three-seed standard deviation and is compared with the
threshold \tailCrit{} of Sec.~\ref{sec:spec}, which is looser than a
prediction interval for one new run. A
positive difference favours the specialist. The specialist wins at $r{=}0.1$,
the single model wins at $r{=}0.3$, and neither separates at $r{=}0.5$ or
$0.7$; what the single model gives that no specialist can is the rest of the
curve.}
\label{tab:specfull}
\end{table}

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{3}{c}{out of domain} & \multicolumn{3}{c}{FFHQ test} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
$r$ & vs.\ input & vs.\ filtered & FID & vs.\ input & vs.\ filtered & FID \\
\midrule
0.05 & \oodOodZeroFive  & \oodOodFZeroFive  & \oodOodFidZeroFive  & \oodIndZeroFive  & \oodIndFZeroFive  & \oodIndFidZeroFive \\
0.10 & \oodOodOneZero   & \oodOodFOneZero   & \oodOodFidOneZero   & \oodIndOneZero   & \oodIndFOneZero   & \oodIndFidOneZero \\
0.20 & \oodOodTwoZero   & \oodOodFTwoZero   & \oodOodFidTwoZero   & \oodIndTwoZero   & \oodIndFTwoZero   & \oodIndFidTwoZero \\
0.30 & \oodOodThreeZero & \oodOodFThreeZero & \oodOodFidThreeZero & \oodIndThreeZero & \oodIndFThreeZero & \oodIndFidThreeZero \\
\bottomrule
\end{tabular}
\caption{\oodN{} general-scene photographs against \oodN{} FFHQ test images, run
with the same script and the prompt ``a high-quality photo''
(Sec.~\ref{sec:ood}). ``Vs.\ input'' correlates the grayscale output with the
grayscale input photograph, ``vs.\ filtered'' with the filtered condition the
model was given; neither filters the output, so these are not the
structure-consistency values of Table~\ref{tab:main}. Each FID is against that
run's own input photographs. The control falls in both; the image quality does
not transfer.}
\label{tab:oodfull}
\end{table}

\begin{table}[!tbp]\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{3}{c}{\sdialN{} face sketches} & \multicolumn{2}{c}{FFHQ test photos} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
$r$ & vs.\ original & $\pm$ SEM & vs.\ filtered & vs.\ original & vs.\ filtered \\
\midrule
0.00 & \sdialZero       & \sdialSemZero       & \sdialFZero       & --- & --- \\
0.05 & \sdialZeroFive   & \sdialSemZeroFive   & \sdialFZeroFive   & \oodIndZeroFive  & \oodIndFZeroFive \\
0.10 & \sdialOneZero    & \sdialSemOneZero    & \sdialFOneZero    & \oodIndOneZero   & \oodIndFOneZero \\
0.20 & \sdialTwoZero    & \sdialSemTwoZero    & \sdialFTwoZero    & \oodIndTwoZero   & \oodIndFTwoZero \\
0.30 & \sdialThreeZero  & \sdialSemThreeZero  & \sdialFThreeZero  & \oodIndThreeZero & \oodIndFThreeZero \\
\midrule
\multicolumn{6}{l}{\itshape beyond the range we claim} \\
0.50 & \sdialFiveZero   & \sdialSemFiveZero   & \sdialFFiveZero   & --- & --- \\
0.70 & \sdialSevenZero  & \sdialSemSevenZero  & \sdialFSevenZero  & --- & --- \\
1.00 & \sdialOneHundred & \sdialSemOneHundred & \sdialFOneHundred & --- & --- \\
\bottomrule
\end{tabular}
\caption{The dial applied to \sdialN{} hand-drawn face sketches (main model,
sampling seed 2026, prompt ``a high-quality photo of a face''). ``Vs.\ original''
correlates the grayscale output with the unfiltered drawing; ``vs.\ filtered''
with the drawing filtered at $r$ that the model was given. Neither filters the
output, so neither is the structure-consistency instrument of
Table~\ref{tab:main}. The photograph columns come from the same script on FFHQ
test images with the prompt ``a high-quality photo'' (Table~\ref{tab:oodfull}).
The $r{=}0$ row is an independent re-run of the zero-shot evaluation of
Figure~\ref{fig:sketch} (\sketchFaceSC{} there). No specialist was run on
sketches.}
\label{tab:sketchspec}
\end{table}

\section{Reproducibility}
\label{sec:repro}

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
Sketch data       & CUHK Face Sketch database, \AUTHORINPUT{A28}; \texttt{rhfeiyang/}\allowbreak\texttt{photo-sketch-pair-500}, \AUTHORINPUT{A29} \\
Optimiser         & AdamW, lr $10^{-4}$, $\beta{=}(0.9,0.999)$, weight decay 0.01, cosine, 500 warm-up steps \\
Batch / precision & 32 / bf16, gradient clipping 1.0, 12 epochs, Min-SNR $\gamma{=}5$, one H200 \\
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
