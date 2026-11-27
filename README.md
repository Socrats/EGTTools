# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Socrats/EGTTools/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                            |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/egttools/\_\_init\_\_.py                                    |       60 |       26 |     57% |8-9, 28-29, 36-51, 56-63, 67-68 |
| src/egttools/\_min\_dependencies.py                             |       16 |       16 |      0% |      6-60 |
| src/egttools/analytical/\_\_init\_\_.py                         |        6 |        2 |     67% |       7-8 |
| src/egttools/analytical/sed\_analytical.py                      |      247 |       21 |     91% |65-66, 97-110, 238-239, 616, 660, 740, 768, 848 |
| src/egttools/analytical/utils.py                                |      120 |       26 |     78% |63, 108, 172, 195, 241-281, 308 |
| src/egttools/behaviors/CPR/abstract\_cpr\_strategy.py           |       23 |        7 |     70% |7, 13, 17, 21, 25, 29, 32 |
| src/egttools/behaviors/CPR/cpr\_strategies.py                   |      139 |       22 |     84% |26-27, 30, 35, 38, 41, 44, 47, 50, 76, 79, 102, 105, 116, 128, 131, 162, 165, 191, 194, 220, 223 |
| src/egttools/behaviors/CRD/\_\_init\_\_.py                      |        8 |        2 |     75% |       6-7 |
| src/egttools/behaviors/CRD/goal\_based.py                       |       26 |       20 |     23% |40-47, 50-65, 68, 72 |
| src/egttools/behaviors/CRD/moving\_average.py                   |       26 |       20 |     23% |39-45, 48-60, 63, 66 |
| src/egttools/behaviors/CRD/time\_based.py                       |       12 |        5 |     58% |17-18, 21, 24, 27 |
| src/egttools/behaviors/NormalForm/TwoActions/\_\_init\_\_.py    |        6 |        2 |     67% |     12-13 |
| src/egttools/behaviors/NormalForm/TwoActions/nfg\_strategies.py |       96 |       66 |     31% |47-51, 54-57, 60, 63, 66, 87-92, 95-111, 114, 117, 120, 133-136, 139-149, 152, 155, 158, 180-185, 188-200, 204, 207, 210 |
| src/egttools/behaviors/NormalForm/\_\_init\_\_.py               |        6 |        2 |     67% |       6-7 |
| src/egttools/behaviors/\_\_init\_\_.py                          |        4 |        0 |    100% |           |
| src/egttools/behaviors/opinion\_behaviors.py                    |       12 |       12 |      0% |     18-35 |
| src/egttools/behaviors/pgg\_behaviors.py                        |       11 |        1 |     91% |        35 |
| src/egttools/datastructures/\_\_init\_\_.py                     |        5 |        2 |     60% |       5-6 |
| src/egttools/distributions/\_\_init\_\_.py                      |        5 |        2 |     60% |      9-10 |
| src/egttools/games/\_\_init\_\_.py                              |        9 |        2 |     78% |     21-22 |
| src/egttools/games/abstract\_games.py                           |       55 |       33 |     40% |57, 73-83, 134-139, 157, 174, 198-204, 207, 210, 213, 216, 219-224, 227-229 |
| src/egttools/games/informal\_risk.py                            |       76 |       57 |     25% |46-54, 58-71, 74-84, 105-118, 121-132, 136, 140, 144, 147-152, 155-162 |
| src/egttools/games/nonlinear\_games.py                          |      177 |       94 |     47% |19-29, 46-57, 60-70, 73-82, 85, 88-96, 102-114, 117-121, 124-127, 131-138, 141-151, 154-164, 167, 170-171, 174, 177-182, 289-298 |
| src/egttools/games/opinion\_game.py                             |       52 |       52 |      0% |    18-117 |
| src/egttools/games/pgg.py                                       |       51 |       10 |     80% |85-96, 102-109 |
| src/egttools/helpers/vectorized.py                              |       19 |       10 |     47% |24-25, 56-57, 88-96 |
| src/egttools/numerical/\_\_init\_\_.py                          |       15 |        2 |     87% |       5-6 |
| src/egttools/numerical/estimators.py                            |      181 |       89 |     51% |66-70, 91-92, 112-113, 118, 122-123, 243, 246, 261-279, 290-304, 315-333, 347-365, 399, 421-422, 425, 439-457, 469-490, 502-542 |
| src/egttools/numerical/indicators.py                            |       25 |        3 |     88% |     80-87 |
| src/egttools/numerical/linear\_operator.py                      |       30 |        0 |    100% |           |
| src/egttools/numerical/mpi/\_\_init\_\_.py                      |        6 |        6 |      0% |     22-29 |
| src/egttools/numerical/results.py                               |      111 |       56 |     50% |35-36, 39-47, 63-64, 67-68, 91-92, 95-105, 123-124, 127-137, 152-155, 179-188, 191-202 |
| src/egttools/numerical/structure/\_\_init\_\_.py                |        6 |        2 |     67% |     11-12 |
| src/egttools/numerical/structure/factories.py                   |       15 |        1 |     93% |       105 |
| src/egttools/plotting/\_\_init\_\_.py                           |        9 |        2 |     78% |     23-24 |
| src/egttools/plotting/barplots.py                               |       82 |       82 |      0% |     1-157 |
| src/egttools/plotting/helpers.py                                |      239 |      119 |     50% |35-41, 168, 207-218, 255-296, 316-322, 351-358, 361-370, 382, 412-458, 474, 486, 519-523, 559, 570, 623 |
| src/egttools/plotting/indicators.py                             |      277 |       51 |     82% |163, 167, 172-227, 280, 303, 602-607 |
| src/egttools/plotting/network\_plots.py                         |      155 |       71 |     54% |47, 95-126, 177, 226-280, 329, 464, 468-478 |
| src/egttools/plotting/simplex2d.py                              |      251 |       74 |     71% |251, 355, 393-409, 448-452, 481-498, 559, 601-612, 620, 748-772, 779, 781, 867-875, 1070-1086 |
| src/egttools/plotting/simplex3d.py                              |      346 |       23 |     93% |42-43, 57, 139, 193, 257, 579, 591, 597, 603, 607, 649, 751, 765, 771, 778, 782, 824, 841, 941, 952, 1190, 1202 |
| src/egttools/plotting/simplified.py                             |      188 |       24 |     87% |57, 91-92, 98, 106, 112, 119, 127, 145, 151, 173, 191, 271, 274, 416, 419, 421, 427, 433, 486, 489, 491, 497, 500 |
| src/egttools/utils.py                                           |      126 |       25 |     80% |65-66, 68-69, 77, 83, 141, 147, 214, 383-384, 423-429, 450-454, 458-462 |
| **TOTAL**                                                       | **3329** | **1142** | **66%** |           |

2 empty files skipped.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Socrats/EGTTools/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Socrats/EGTTools/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Socrats/EGTTools/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Socrats/EGTTools/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FSocrats%2FEGTTools%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Socrats/EGTTools/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.