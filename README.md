How to start the web app
----------------------------
1. Clone on your Host machine: 
2. Open the repo in VS Code and in a Dev Container. if successful, you should see: `/workspaces/hcc $`
3. In your dev container's terminal, run the following commands:
`python app.py` 
4. You will be prompted to open the localhost in browser
5. From your browser, upload the cvs file and complete the task.

Task
------------------------------
For each entry, which one of the predicate options best connects the subject and object solely based on the summary? 
- Do not use any other external knowledge except what you can see or infer from the summary. 
- Make sure the subject and objects are mentioned in the abstract. If they are not, then None will be the best fit
- Once you select an option, remeber to `submit classification` before moving on to the `next` entry






About the Dataset
------------------------------

There are:
    -   7745 healpaca relationships out of which 3370 is unique
    -   30 of the unique 3370 healpaca relationships is biolink standard. 3340 is not
Sample relationship frquencies:
| Relationship                        | Count |
|------------------------------------|-------|
| associated with                    | 304   |
| causes                             | 210   |
| reduces                            | 164   |
| treats                             | 127   |
| increases                          | 106   |
| have                               | 98    |
| induces                            | 84    |
| use                                | 73    |
| leads to                           | 69    |
| affects                            | 65    |
| treated with                       | 63    |
| has                                | 61    |
| decreases                          | 55    |
| inhibits                           | 52    |
| predicts                           | 49    |
| prevents                           | 47    |
| improves                           | 46    |
| contributes to                     | 46    |
| affected by                        | 46    |
| experience                         | 45    |
| found in                           | 43    |
| activates                          | 40    |
| involved in                        | 39    |
| influences                         | 35    |
| used for treatment                 | 35    |
| treated by                         | 34    |
| increases risk of                  | 33    |
| experienced                        | 33    |
| had                                | 32    |
| used for                           | 32    |
| regulates                          | 30    |
| received                           | 28    |
| used for treating                  | 28    |
| reverses                           | 24    |
| not associated with                | 24    |
| detects                            | 24    |
| contribute to                      | 24    |
| used in treatment                  | 24    |
| measures                           | 24    |
| attenuates                         | 23    |
| promotes                           | 22    |
| modulates                          | 22    |
| cause                              | 21    |
| expressed in                       | 21    |
| used in                            | 21    |
| diagnosed with                     | 20    |
| targets                            | 20    |
| used                               | 20    |
| impairs                            | 19    |
| receive                            | 19    |
| produces                           | 18    |
| developed                          | 17    |
| suppresses                         | 17    |
| used in treatment of               | 16    |
| mediates                           | 15    |
| used to treat                      | 15    |
| prescribed                         | 15    |
| blocks                             | 14    |
| identified in                      | 14    |
| decreased                          | 14    |
| increased                          | 14    |
| enhances                           | 14    |
| impacts                            | 14    |
| treat                              | 14    |
| reduce                             | 14    |
| is associated with                 | 13    |
| alters                             | 13    |
| undergo                            | 13    |
| develop                            | 12    |
| includes                           | 12    |
| linked to                          | 11    |
| caused by                          | 11    |
| exhibit                            | 11    |
| upregulates                        | 11    |
| treated                            | 11    |
| reduced                            | 11    |
| misuse                             | 11    |
| prescribed after                   | 11    |
| responds to                        | 10    |
| identified                         | 10    |
| may cause                          | 10    |
| protects against                   | 10    |
| binds to                           | 10    |
| interacts with                     | 10    |
| prescribe                          | 10    |
| engage in                          | 10    |
| initiated                          | 10    |
| affected                           | 9     |
| stimulates                         | 9     |
| increases expression of            | 9     |
| improved                           | 9     |
| correlated with                    | 9     |
| used to study                      | 9     |
| addresses                          | 9     |
| drives                             | 9     |
| have higher prevalence of          | 9     |
| at risk of                         | 9     |
| caused                             | 8     |
| receiving                          | 8     |
| expresses                          | 8     |
| predictor of                       | 8     |
| prescribed to                      | 8     |
| have higher odds of                | 8     |
| associated with increased consumption | 8  |
| influence                          | 8     |
| at risk for                        | 8     |
| used by                            | 8     |
| more likely to have                | 8     |
| positively associated with         | 8     |
| increases levels of                | 7     |
| screened for                       | 7     |
| requires                           | 7     |
| disrupts                           | 7     |
| protects                           | 7     |
| contains                           | 7     |
| located in                         | 7     |
| associated with lower              | 7     |
| provides                           | 7     |
| influenced by                      | 7     |
| prescribing                        | 7     |
| used to assess                     | 7     |
| uses                               | 7     |
| relieves                           | 7     |
| more likely to use                 | 7     |
| exposed to                         | 7     |
| discriminates risk use             | 7     |
| less likely to receive             | 7     |

*...Table continues ...*

