# Datasets

Candidate datasets for training classification heads on frozen HeAR embeddings (2 s clips, 16 kHz).

Collected 2026-11-04 from official pages, papers and repos. `?` = not verified.

## Used by the HeAR authors

Google publishes no "recommended for fine-tuning" list. The model card ([HAI-DEF](https://developers.google.com/health-ai-developer-foundations/hear/model-card), [HF google/hear](https://huggingface.co/google/hear)) lists six evaluation datasets, the same ones benchmarked in the [HeAR paper](https://arxiv.org/abs/2403.02522). The Vertex AI Model Garden page needs a login and was not checked.

| Dataset | What it is | HeAR tasks | Access, license |
|---|---|---|---|
| [FSD50K](https://zenodo.org/records/4060432) | Freesound sound events, 51k clips, 200 classes | event detection: breathing, cough, laughter, respiratory sounds, sneeze, speech | public, CC BY (per-clip CC mix) |
| [FluSense](https://github.com/Forsad/FluSense-data) | AudioSet subset re-annotated for cough-like events | event detection: breathe, cough, gasp, sneeze, sniffle, speech, throat-clear | annotations on GitHub, audio separate; license ? |
| [CoughVID](https://zenodo.org/records/7024894) | crowdsourced coughs (EPFL) | COVID status, sex | public, CC BY 4.0 |
| [Coswara](https://zenodo.org/records/7188627) | cough, breathing, vowels, counting (IISc) | COVID, sex, smoking, age | public, CC BY 4.0 |
| [CIDRZ](https://www.kaggle.com/datasets/googlehealthai/google-health-ai) | TB-clinic coughs in Zambia, 599 patients, 3 phone models | TB, sex, smoking, age, BMI, CXR findings; device generalization | "proprietary" in the paper, later on Kaggle; terms ? |
| [SpiroSmart](https://dl.acm.org/doi/10.1145/2370216.2370261) | forced exhalations into a phone, COPD clinic | FEV1, FVC, FEV1/FVC, PEF, FET, sex | no public dataset found |

Pretraining used YT-NS: 313M 2 s clips cut from public YouTube videos by a health acoustic event detector. Not released.

## Cough / breath / voice (same domain as HeAR)

| Dataset | Sound | Size | Labels | Subject IDs | Access, license |
|---|---|---|---|---|---|
| [COUGHVID v3](https://zenodo.org/records/7024894) (downloaded: `make download-coughvid`) | cough | 34,434 recordings | experts on 2,890 recordings (130 rated by all 4 physicians): quality, cough_type wet/dry, dyspnea, wheezing, stridor, choking, congestion, diagnosis, severity; self-reported status, age, gender, respiratory_condition, fever_muscle_pain on ~60% | no, recording UUID only | Zenodo, CC BY 4.0 |
| [Coswara](https://github.com/iiscleap/Coswara-Data) | cough, breathing, vowels, counting (9 types) | 2,635 subjects, ~23.7k recordings | self-reported: COVID (RAT/RT-PCR), asthma, chronic lung disease, pneumonia, symptoms, smoker; manual audio quality 0–2 | yes | GitHub/Zenodo, CC BY 4.0 |
| [UK COVID-19 Vocal Audio](https://zenodo.org/records/10043978) (UKHSA) | cough, three-cough, exhalation | 72,999 participants | SARS-CoV-2 PCR; influenza PCR for 27%; self-reported symptoms, asthma, respiratory conditions | yes, ready train/test splits | open version on Zenodo, OGL v3 (coarsened metadata); full version on request |
| [COVID-19 Sounds](https://github.com/cam-mobsys/covid19-sounds-neurips) (Cambridge) | cough, breathing, voice | 53,449 samples, 36,116 participants | self-reported: COVID test, symptoms incl. dry/wet cough, medical history, smoking | yes | data transfer agreement (academic) |
| [CODA TB DREAM](https://www.synapse.org/TBcough) | solicited 0.5 s coughs + longitudinal | 2,143 adults, 7 countries; open train 1,105 | TB by sputum Xpert Ultra + culture; clinical metadata | yes | Synapse, controlled access; hidden test set |
| [Sound-Dr](https://github.com/ReML-AI/Sound-Dr) | cough, mouth and nose breathing | ~1,310 recordings | self-reported symptoms incl. wet/dry cough, stuffy nose; COVID | no | Google Drive (password in repo); license ? |
| [COPD-Cough](https://zenodo.org/records/10209837) | phone-recorded cough | 221 COPD, 632 healthy recordings | COPD vs healthy, clinician-endorsed (protocol ?) | ? | Zenodo, CC BY 4.0 |
| [Pediatric bronchitis vs pneumonia](https://doi.org/10.6084/m9.figshare.21176197) | cough with hospital noise | 173 children, 1 recording each | clinical: 82 pneumonia, 91 bronchitis | 1 recording = 1 subject | Figshare, license ? |
| [Virufy](https://github.com/virufy/virufy-data) | cough | 16 patients clinical + crowdsourced subset | COVID PCR | yes | GitHub, "Other" |
| ESC-50 `coughing`, AudioSet `Cough` | cough as an event class | small | "is cough" only | no | negatives / cough detection, not disease |

## Lung auscultation (stethoscope)

Contact-microphone recordings, often narrow-band: strong domain shift from HeAR's air-microphone pretraining. Secondary use only.

| Dataset | Size | Labels | Access, license |
|---|---|---|---|
| [ICBHI 2017](https://bhichallenge.med.auth.gr) | 920 recordings, 126 subjects | crackles/wheezes per cycle; diagnosis: COPD, URTI, LRTI, asthma, pneumonia, bronchiectasis, bronchiolitis | free for research, official split |
| [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) | 2,683 recordings, 292 children | 11 pediatricians: wheeze, stridor, rhonchi, crackles | GitHub, CC BY 4.0 |
| [HF_Lung_V1](https://gitlab.com/techsupportHF/HF_Lung_V1) | 9,765 × 15 s | wheeze, stridor, rhonchi, crackle; breath phases | GitLab, license ?; 4 kHz only |
| [KAUH](https://doi.org/10.17632/jwyy9np4gv.3) | 112 subjects | asthma, COPD, pneumonia, bronchiectasis, heart failure, fibrosis | Mendeley Data, license ? |
| [RespiratoryDatabase@TR](https://doi.org/10.17632/p9z4h98s6j.2) | 75 subjects, 12 channels | COPD severity GOLD 0–4 | Mendeley Data, license ? |

Not public: pertussis/croup clinical sets (IndiCough2024), asthma cough cohorts, Swaasa.

## Coverage of head targets

| Target | Sources |
|---|---|
| wet / dry cough | COUGHVID (experts); self-reported in Sound-Dr, COVID-19 Sounds |
| wheezing, stridor, congestion, dyspnea in cough | COUGHVID only; wheeze/stridor in breath sounds: SPRSound, HF_Lung, ICBHI |
| asthma / COPD | UK COVID-19 Vocal, Coswara, COPD-Cough; auscultation: ICBHI, KAUH, RespiratoryDatabase@TR |
| TB | CODA TB DREAM, CIDRZ |
| pneumonia / URTI / LRTI | Coswara (self-reported), pediatric bronchitis vs pneumonia; auscultation: ICBHI |
| healthy vs symptomatic | UK COVID-19 Vocal, Coswara, COVID-19 Sounds |

Most useful next: UK COVID-19 Vocal Audio (scale, subject splits, PCR labels, open), Coswara (subject IDs, respiratory flags, direct download), CODA TB (lab-confirmed labels, needs access request).
