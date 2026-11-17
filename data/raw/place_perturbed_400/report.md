# Dataset report: 400 episodes

- success: 303/400 (75.8 %)
- length: {'min': 8.57, 'mean': 14.59, 'max': 20.0} s (successful: {'min': 8.57, 'mean': 13.76, 'max': 20.0})
- frames: 175038, disk: 29.5 MB (0.07 MB per episode)
- expert grasp attempts: {1: 173, 2: 133, 3: 94}

**task**: place/failure 97, place/success 303

**object**: blue_can/failure 31, blue_can/success 58, green_cube/failure 20, green_cube/success 72, red_cube/failure 17, red_cube/success 87, yellow_ball/failure 29, yellow_ball/success 86

**target**: purple_tray/failure 47, purple_tray/success 145, white_tray/failure 50, white_tray/success 158

**arm**: left/failure 50, left/success 142, right/failure 47, right/success 161

**perturbation**: drop/failure 72, drop/success 52, nudge/failure 14, nudge/success 162, slip/failure 11, slip/success 89

**instruction_template**: 0 98, 1 107, 2 101, 3 94

**failure reasons**

- blue_can is beside the purple_tray: 17
- yellow_ball is beside the white_tray: 17
- blue_can is beside the white_tray: 14
- yellow_ball is beside the purple_tray: 12
- green_cube is beside the purple_tray: 12
- red_cube is beside the white_tray: 11
- green_cube is beside the white_tray: 8
- red_cube is beside the purple_tray: 6

**action slots that never move** (expected: the base, the head, the idle arm's joints when only one arm is ever used): base_vx, base_vy, base_wz
