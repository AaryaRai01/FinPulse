# FinPulse India — ML-focused frontend

A React + Vite frontend for the FinPulse India academic Gaussian HMM project.

## Run locally

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## Pages

1. Overview — state distribution, annual composition, emission profiles and transition matrix.
2. Case study — CPHS research context, variables and limitations.
3. ML training & implementation — sequence preparation, scaling, Gaussian HMM parameterisation, Baum–Welch/EM, convergence, Viterbi decoding, Forward–Backward posterior probabilities, model diagnostics and a household sequence explorer.
4. Results explorer — filter saved household-year inference records.

## Data files

- `public/data/model_summary.json` — model configuration and aggregated results.
- `public/data/household_hidden_states.csv` — 21,679 household-year inference records, including Viterbi states and posterior probabilities.

## Model scope

The tied-covariance Gaussian HMM was trained using 2,000 sampled sequence segments (4,436 observations), converged in 49 iterations, and was applied to 21,679 eligible household-year observations.

The browser displays saved training and inference outputs. It does not retrain the model. No iteration-by-iteration likelihood history was exported, so the UI intentionally does not invent a training-loss curve.

The dataset does not provide monthly repayment histories or explicit loan-default labels. The project does not predict loan default. Hidden states are exploratory model-inferred profiles, not independently verified poverty classifications.
