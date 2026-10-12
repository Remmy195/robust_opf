# Case data

Only `case_ACTIVSg200.m` ships here. Both families below come from the Electric
Grid Test Case Repository, https://electricgrids.engr.tamu.edu. Save a case in
`data/` under the name in the table; `result.json` records the SHA-256 of the
case it ran.

## EPIGRIDS

The paper's frontier systems.

| File | From the download | SHA-256 |
|---|---|---|
| `case_EPINewEngland.m` | `New England/MatPower Cases/Medium/Winter68Tight.m` | `9a01381c3557df10825557f01d8f1cceb06df28ba75f3e6851dbe7e69b551930` |
| `case_EPIWisconsin.m` | `Wisconsin/Wisconsin_1664.m` | `039df3ac7fcee9add540f99462a9d2667438a6adb7f30b1caf98ace5adbca449` |
| `case_EPIFlorida.m` | `Florida/Base_Florida_42GW.m` | `677a12456333ae3f62779aea1da7e029abb8379478aad0e6c9f15bf1f52ae930` |
| `case_EPIWestern.m` | `Western Network/Base_West_Interconnect_121GW.m` | `5545bf06993a5f5735261896f361e631453392239bbdca4f619f732c078bae27` |
| `case_EPIEastern.m` | `Eastern Network/Base_Eastern_Interconnect_515GW.m` | `5becfa3af0d4e295b7ab259ab62abb0167b7c9e75a02c56798965a169d70abb7` |

Cite: J. M. Snodgrass, "Tractable Algorithms for Constructing Electric Power
Network Models," PhD thesis, University of Wisconsin-Madison, 2021.

## ACTIVSg

The paper's counterfactual validation uses ACTIVSg10k and ACTIVSg25k.

| File | SHA-256 |
|---|---|
| `case_ACTIVSg200.m` | `3c92cb217e1e04bb764d2566ccf01f3f2e2ac8af2d6b2907b0619ee335165c87` |
| `case_ACTIVSg500.m` | `bf7c4831dd4a1e4f022ae8f1681bc76622422e1057c8433f6f0056ed18224a83` |
| `case_ACTIVSg2000.m` | `5edb60e97153c27de68403499174557213090a5b04da1bbc34b3196715c5da94` |
| `case_ACTIVSg10k.m` | `78b1c51c575a5356d41529e1b7cc5d532457ccaed009cec35c1dbb46fd45482f` |
| `case_ACTIVSg25k.m` | `a4157af7ff1c2f9b73fc1132a80fbdf97659912050ffb35f9259d7e057954b0e` |
| `case_ACTIVSg70k.m` | `22d815497ccb587d3b7d167c740e8e75a7d509d813c34c186d08dec77f757e08` |

Copyright (c) 2017-2018 A.B. Birchfield, T. Xu, K.M. Gegner, K.S. Shetye and
T.J. Overbye. Creative Commons Attribution 4.0 International,
http://creativecommons.org/licenses/by/4.0/

Cite:

1. A.B. Birchfield, T. Xu, K.M. Gegner, K.S. Shetye, T.J. Overbye, "Grid
   Structural Characteristics as Validation Criteria for Synthetic Networks,"
   IEEE Transactions on Power Systems, vol. 32, no. 4, pp. 3258-3265, July 2017.
   DOI: 10.1109/TPWRS.2016.2616385
2. For ACTIVSg70k, also: A.B. Birchfield, T. Xu, T.J. Overbye, "Power Flow
   Convergence and Reactive Power Planning in the Creation of Large Synthetic
   Grids," IEEE Transactions on Power Systems, 2018.
   DOI: 10.1109/TPWRS.2018.2813525
