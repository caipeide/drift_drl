#!/bin/sh
echo '########  Testing: DDPG ########'
python test_ddpg.py

echo '########  Testing: DQN ########'
python test_dqn.py

echo '########  Testing: SAC ########'
python test_sac.py --model sac-stg2

echo '########  Testing: SAC-WOS ########'
python test_sac.py --model sac-wos

echo '########  Testing: SAC on different vehicles ########'
python test_sac_different_vehicles.py --vehicleNum 2

python test_sac_different_vehicles.py --vehicleNum 3

python test_sac_different_vehicles.py --vehicleNum 4

echo '########  Testing: SAC on vehicle-1 with different frictions ########'
python test_sac_different_frictions.py

echo '########  Testing: SAC with rough reference trajectory ( x,y waypoints only) ########'
python test_sac_app.py