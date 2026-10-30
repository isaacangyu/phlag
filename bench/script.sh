# # dominant and alternative topologies
# benchmark -w 50-50k --np null-alt-num --floor-fix 
# benchmark -w 50-50k --np null-alt-num --floor-fix --rho 0.9 --beta 4
# benchmark -w 50-50k --floor-fix # free, rerun with lower floor
# benchmark -w 50-50k --np null-alt-equal --floor-fix --null-majority
# benchmark -w 50-50k --np null-alt-equal --floor-fix --null-majority --rho 0.9 --beta 4
# benchmark -w 50-50k --np null-alt-equal --floor-fix # rerun just alternative with lower floor
# benchmark -w 50-50k --np null-alt-equal --floor-fix --rho 0.9 --beta 4 # rerun just alternative with lower floor


benchmark -w 50-50k --np null-alt-num --floor-fix --null-majority
benchmark -w 50-50k --np null-alt-num --floor-fix --null-majority --rho 0.9 --beta 4
benchmark -w 50-50k --np null-alt-equal --no-floor
benchmark -w 50-50k --np null-alt-equal --no-floor --rho 0.9 --beta 4
benchmark -w 50-50k --np null-alt-equal --no-floor --null-majority
benchmark -w 50-50k --np null-alt-equal --no-floor --null-majority --rho 0.9 --beta 4
benchmark -w 50-50k --no-floor
benchmark -w 50-50k --no-floor --rho 0.9 --beta 4