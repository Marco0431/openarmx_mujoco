"""平放版扰动窗口，指定平放入口及默认搬运距离。"""
from pathlib import Path
from run_handheld_disturbance import main

if __name__ == '__main__':
    raise SystemExit(main(runtime=Path(__file__).resolve().parent / 'exp2_nmpcc_latest.py',
                         scene_label='平放', output_name='flat_disturbance', default_distance_cm='10'))
