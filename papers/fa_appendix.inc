\appendix
\section{Full per-cutoff results}

The main text reports each experiment at the cutoff that carries its argument.
The tables here give every cutoff that was measured, from the same
\texttt{results.json} files the rest of the paper is generated from.

\begin{table}[h]\centering\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrrr}
\toprule
variant & $r{=}0$ & 0.05 & 0.1 & 0.3 & 0.5 \\
\midrule
working model        & \scMainZero      & \scMainZeroFive      & \scMainOneZero      & \scMainThreeZero      & \scMainFiveZero \\
$1{\times}1$ window  & \ablKoneZero     & \ablKoneZeroFive     & \ablKoneOneZero     & \ablKoneThreeZero     & \ablKoneFiveZero \\
$5{\times}5$ window  & \ablKfiveZero    & \ablKfiveZeroFive    & \ablKfiveOneZero    & \ablKfiveThreeZero    & \ablKfiveFiveZero \\
64-ch stem           & \ablStemsfZero   & \ablStemsfZeroFive   & \ablStemsfOneZero   & \ablStemsfThreeZero   & \ablStemsfFiveZero \\
256-ch stem          & \ablStemskipZero & \ablStemskipZeroFive & \ablStemskipOneZero & \ablStemskipThreeZero & \ablStemskipFiveZero \\
average downsample   & \ablDavgZero     & \ablDavgZeroFive     & \ablDavgOneZero     & \ablDavgThreeZero     & \ablDavgFiveZero \\
colour condition     & \ablColorZero    & \ablColorZeroFive    & \ablColorOneZero    & \ablColorThreeZero    & \ablColorFiveZero \\
log cutoff sampler   & \ablLogZero      & \ablLogZeroFive      & \ablLogOneZero      & \ablLogThreeZero      & \ablLogFiveZero \\
sqrt cutoff sampler  & \ablSqrtZero     & \ablSqrtZeroFive     & \ablSqrtOneZero     & \ablSqrtThreeZero     & \ablSqrtFiveZero \\
no condition dropout & \ablDropZero     & \ablDropZeroFive     & \ablDropOneZero     & \ablDropThreeZero     & \ablDropFiveZero \\
no zero-init gate    & \ablNogateZero   & \ablNogateZeroFive   & \ablNogateOneZero   & \ablNogateThreeZero   & \ablNogateFiveZero \\
\midrule
\multicolumn{6}{l}{\itshape superseded designs, kept for the record} \\
256px source pipeline & --- & \scEresZeroFive & \scEresOneZero & \scEresThreeZero & --- \\
VAE condition         & --- & \scEbaseZeroFive & \scEbaseOneZero & \scEbaseThreeZero & --- \\
\bottomrule
\end{tabular}
\caption{Structure consistency for every one-field ablation, at every measured
cutoff. Each row changes one field of the working model and nothing else. The
last block holds two designs the paper argues against: building the condition at
256 and upsampling (FID \fidEres{} at $r{=}0.1$), and reading it through the
frozen VAE.}
\label{tab:ablfull}
\end{table}

\begin{table}[h]\centering\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrrr}
\toprule
variant & $r{=}0$ & 0.05 & 0.1 & 0.3 & 0.5 \\
\midrule
working model          & \fidMainZero & \fidMainZeroFive & \fidMainOneZero & \fidMainThreeZero & \fidMainFiveZero \\
$1{\times}1$ window    & \ablFidKoneZero & \ablFidKoneZeroFive & \ablFidKoneOneZero & \ablFidKoneThreeZero & \ablFidKoneFiveZero \\
$5{\times}5$ window    & \ablFidKfiveZero & \ablFidKfiveZeroFive & \ablFidKfiveOneZero & \ablFidKfiveThreeZero & \ablFidKfiveFiveZero \\
64-ch stem             & \ablFidStemsfZero & \ablFidStemsfZeroFive & \ablFidStemsfOneZero & \ablFidStemsfThreeZero & \ablFidStemsfFiveZero \\
256-ch stem            & \ablFidStemskipZero & \ablFidStemskipZeroFive & \ablFidStemskipOneZero & \ablFidStemskipThreeZero & \ablFidStemskipFiveZero \\
average downsample     & \ablFidDavgZero & \ablFidDavgZeroFive & \ablFidDavgOneZero & \ablFidDavgThreeZero & \ablFidDavgFiveZero \\
colour condition       & \ablFidColorZero & \ablFidColorZeroFive & \ablFidColorOneZero & \ablFidColorThreeZero & \ablFidColorFiveZero \\
log cutoff sampler     & \ablFidLogZero & \ablFidLogZeroFive & \ablFidLogOneZero & \ablFidLogThreeZero & \ablFidLogFiveZero \\
sqrt cutoff sampler    & \ablFidSqrtZero & \ablFidSqrtZeroFive & \ablFidSqrtOneZero & \ablFidSqrtThreeZero & \ablFidSqrtFiveZero \\
no condition dropout   & \ablFidDropZero & \ablFidDropZeroFive & \ablFidDropOneZero & \ablFidDropThreeZero & \ablFidDropFiveZero \\
no zero-init gate      & \ablFidNogateZero & \ablFidNogateZeroFive & \ablFidNogateOneZero & \ablFidNogateThreeZero & \ablFidNogateFiveZero \\
\bottomrule
\end{tabular}
\caption{FID for the same one-field ablations as Table~\ref{tab:ablfull}. No
variant separates from the working model by more than the sampling spread, so
the ablations move structure consistency without trading image quality for it.}
\label{tab:ablfid}
\end{table}

\begin{table}[h]\centering\footnotesize
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
\caption{Every metric for the working model, at every cutoff. The SC and FID
columns are the ones plotted in Figure~\ref{fig:curve}.}
\label{tab:allmetrics}
\end{table}

\begin{table}[h]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
$r$ & skip only & skip only FID & seed mean & seed s.d. \\
\midrule
0.00 & \skipOnlyZero      & \skipOnlyFidZero      & ---               & --- \\
0.05 & \skipOnlyZeroFive  & \skipOnlyFidZeroFive  & \seedMeanZeroFive & \seedStdZeroFive \\
0.10 & \skipOnlyOneZero   & \skipOnlyFidOneZero   & \seedMeanOneZero  & \seedStdOneZero \\
0.20 & \skipOnlyTwoZero   & \skipOnlyFidTwoZero   & \seedMeanTwoZero  & \seedStdTwoZero \\
0.30 & \skipOnlyThreeZero & \skipOnlyFidThreeZero & \seedMeanThreeZero& \seedStdThreeZero \\
0.50 & \skipOnlyFiveZero  & \skipOnlyFidFiveZero  & \seedMeanFiveZero & \seedStdFiveZero \\
0.70 & \skipOnlySevenZero & \skipOnlyFidSevenZero & \seedMeanSevenZero& \seedStdSevenZero \\
1.00 & \skipOnlyOneHundred& \skipOnlyFidOneHundred& ---               & --- \\
\bottomrule
\end{tabular}
\caption{Left: the skip path trained alone (Sec.~\ref{sec:skiponly}). Right:
mean and standard deviation of structure consistency over seeds 2026, 2027 and
2028 for the working model.}
\label{tab:skipseed}
\end{table}

\begin{table}[h]\centering\footnotesize
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
1.00    & \scMainOneZero & \fidMainOneZero & \seedMeanOneZero \\
\bottomrule
\end{tabular}
\caption{The strength knob at $r{=}0.1$, swept finely. Every point pays for
structure consistency with FID; the cutoff dial reaches the same range while FID
stays in the fifties (Table~\ref{tab:allmetrics}). Three points were repeated on
a second seed.}
\label{tab:strengthfull}
\end{table}

\begin{table}[h]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
$r$ & specialist & one model & difference & $z$ \\
\midrule
0.10 & \specOneZero   & \scMainOneZero   & \specDOneZero   & \specZOneZero \\
0.30 & \specThreeZero & \scMainThreeZero & \specDThreeZero & \specZThreeZero \\
0.50 & \specFiveZero  & \scMainFiveZero  & \specDFiveZero  & \specZFiveZero \\
0.70 & \specSevenZero & \scMainSevenZero & \specDSevenZero & \specZSevenZero \\
\bottomrule
\end{tabular}
\caption{Per-cutoff specialists against the single randomised-$r$ model, each at
its own cutoff. A positive difference favours the specialist. The single model
wins at $r{=}0.3$ and loses at $r{=}0.1$; what it gives that no specialist can
is the rest of the curve.}
\label{tab:specfull}
\end{table}

\begin{table}[h]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
$r$ & OOD SC & OOD FID & in-domain SC & in-domain FID \\
\midrule
0.05 & \oodOodZeroFive  & \oodOodFidZeroFive  & \oodIndZeroFive  & \oodIndFidZeroFive \\
0.10 & \oodOodOneZero   & \oodOodFidOneZero   & \oodIndOneZero   & \oodIndFidOneZero \\
0.20 & \oodOodTwoZero   & \oodOodFidTwoZero   & \oodIndTwoZero   & \oodIndFidTwoZero \\
0.30 & \oodOodThreeZero & \oodOodFidThreeZero & \oodIndThreeZero & \oodIndFidThreeZero \\
\bottomrule
\end{tabular}
\caption{Out-of-domain photographs against in-domain, scored against the input
drawing with the same script and prompt (Sec.~\ref{sec:ood}). The control falls
in both; the image quality does not transfer.}
\label{tab:oodfull}
\end{table}

\begin{table}[h]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}
\toprule
$r$ & sketch dial SC & $\pm$ s.e.m. & specialist \\
\midrule
0.00 & \sdialZero       & \sdialSemZero       & --- \\
0.05 & \sdialZeroFive   & \sdialSemZeroFive   & --- \\
0.10 & \sdialOneZero    & \sdialSemOneZero    & \specOneZero \\
0.20 & \sdialTwoZero    & \sdialSemTwoZero    & --- \\
0.30 & \sdialThreeZero  & \sdialSemThreeZero  & \specThreeZero \\
0.50 & \sdialFiveZero   & \sdialSemFiveZero   & \specFiveZero \\
0.70 & \sdialSevenZero  & \sdialSemSevenZero  & \specSevenZero \\
1.00 & \sdialOneHundred & \sdialSemOneHundred & --- \\
\bottomrule
\end{tabular}
\caption{Left: the dial applied to \sdialN{} hand-drawn sketches, scored against
the drawing. Right: per-cutoff specialists at their own cutoff, for the four
cutoffs where one was trained.}
\label{tab:sketchspec}
\end{table}
