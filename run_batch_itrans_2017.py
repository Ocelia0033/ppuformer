# -*- coding: utf-8 -*-
"""批量跑 iTransformer pv2017: pred_len = 1, 4, 8, 24"""

import run

for pl in [1, 4, 8, 24]:
    print("=" * 60)
    print(f"  iTransformer | pv2017 | pred_len={pl}")
    print("=" * 60)
    run.pred_len = pl
    run.dataset_name = "pv2017"
    run.year = None
    run.main()
    print()

print("All done!")
