## Simulate solutions for unit tests in Torsten
rm(list = ls())
gc()

# .libPaths("~/svn-StanPmetrics/script/lib")
library(mrgsolve)
library(dplyr)

## Simulate ME-2 plasma concentrations and ANC values
## for a simplified Friberg-Karlsson model
code <- '
$PARAM CL = 10, Q = 15, VC = 35, VP = 105, KA = 2.0, MTT = 125, 
Circ0 = 5, alpha = 3E-4, gamma = 0.17

$SET delta=0.1 // simulation grid

$CMT GUT CENT PERI PROL TRANSIT CIRC

$MAIN

// Reparametrization
double k10 = CL / VC;
double k12 = Q / VC;
double k21 = Q / VP;
double ktr = 4/MTT;

$ODE 
dxdt_GUT = -KA * GUT;
dxdt_CENT = KA * GUT - (k10 + k12) * CENT + k21 * PERI;
dxdt_PERI = k12 * CENT - k21 * PERI;
dxdt_PROL = ktr * (PROL + Circ0) * ((1 - alpha * CENT/VC) * pow(Circ0/(CIRC + Circ0),gamma) - 1);
dxdt_TRANSIT = ktr * (PROL - TRANSIT);
dxdt_CIRC = ktr * (TRANSIT - CIRC);
'

mod <- mread("acum", tempdir(), code)
e1 <- ev(amt = 10000)
mod %>% ev(e1) %>% mrgsim(end = 500) %>% plot # plot data

## save some data for unit tests (see amounts at t = 1 hour, with no noise)
time <- seq(from = 0.25, to = 2, by = 0.25)
time <- c(time, 4)
xdata <- mod %>% ev(e1) %>% mrgsim(Req = "GUT, CENT, PERI, PROL, TRANSIT, CIRC",
                                   end = -1, add = time,
                                   rescort = 3) %>% as.data.frame
xdata
# ID time          GUT     CENT      PERI          PROL       TRANSIT          CIRC
# 1   1 0.00 10000.000000    0.000    0.0000  0.0000000000  0.000000e+00  0.000000e+00
# 2   1 0.25  6065.306597 3579.304  212.1623 -0.0006874417 -1.933282e-06 -3.990995e-09
# 3   1 0.50  3678.794412 5177.749  678.8210 -0.0022297559 -1.318812e-05 -5.608541e-08
# 4   1 0.75  2231.301599 5678.265 1233.3871 -0.0041121287 -3.824790e-05 -2.508047e-07
# 5   1 1.00  1353.352829 5597.489 1787.0134 -0.0060546255 -7.847821e-05 -7.039447e-07
# 6   1 1.25   820.849983 5233.332 2295.7780 -0.0079139199 -1.335979e-04 -1.533960e-06
# 7   1 1.50   497.870681 4753.865 2741.1870 -0.0096246889 -2.025267e-04 -2.852515e-06
# 8   1 1.75   301.973832 4250.712 3118.6808 -0.0111649478 -2.838628e-04 -4.760265e-06
# 9   1 2.00   183.156387 3771.009 3430.9355 -0.0125357580 -3.761421e-04 -7.345522e-06
# 10  1 4.00     3.354626 1601.493 4374.6747 -0.0192607813 -1.370742e-03 -5.951920e-05

e1 <- ev(amt = 12000, rate = 12000, addl = 14, ii = 12)
mod %>% ev(e1) %>% mrgsim(end = 500) %>% plot # plot data

## save some data for unit tests (see amounts at t = 1 hour, with no noise)
time <- seq(from = 0.25, to = 2, by = 0.25)
time <- c(time, 4)
xdata <- mod %>% ev(e1) %>% mrgsim(Req = "GUT, CENT, PERI, PROL, TRANSIT, CIRC",
                                   end = -1, add = time,
                                   rescort = 3) %>% as.data.frame
xdata
# ID time        GUT      CENT       PERI          PROL       TRANSIT          CIRC
# 1   1 0.00    0.00000    0.0000    0.00000  0.0000000000  0.000000e+00  0.000000e+00
# 2   1 0.25 2360.81604  601.5528   22.49548 -0.0000726505 -1.499109e-07 -2.448992e-10
# 3   1 0.50 3792.72335 1951.4716  152.31877 -0.0004967093 -2.110375e-06 -7.029932e-09
# 4   1 0.75 4661.21904 3599.5932  438.32811 -0.0014439180 -9.454212e-06 -4.809775e-08
# 5   1 1.00 5187.98830 5301.0082  892.11236 -0.0029697950 -2.658409e-05 -1.833674e-07
# 6   1 1.25 3146.67402 6328.6148 1483.47278 -0.0049954464 -5.788643e-05 -5.079766e-07
# 7   1 1.50 1908.55427 6478.2514 2110.88020 -0.0072059086 -1.060159e-04 -1.145673e-06
# 8   1 1.75 1157.59667 6180.6685 2705.53777 -0.0093809310 -1.713345e-04 -2.230652e-06
# 9   1 2.00  702.11786 5681.5688 3235.76051 -0.0114135989 -2.529409e-04 -3.893275e-06
# 10  1 4.00   12.85974 2316.1858 5156.18535 -0.0216237650 -1.312811e-03 -4.956987e-05


e1 <- ev(amt = 1200, ii = 12, ss = 1, cmt = 1)
mod %>% ev(e1) %>% mrgsim(end = 500) %>% plot # plot data
time <- seq(from = 1, to = 10, by = 1)
xdata <- mod %>% ev(e1) %>% mrgsim(Req = "GUT, CENT, PERI, PROL, TRANSIT, CIRC",
                                   end = -1, add = time,
                                   rescort = 3) %>% as.data.frame
xdata
# ID time          GUT     CENT      PERI        PROL     TRANSIT        CIRC
# 1   1    0 1.200000e+03 179.9494  835.4153 -0.08689426 -0.08765832 -0.08765770
# 2   1    1 1.624023e+02 842.7123 1008.6670 -0.08737445 -0.08763983 -0.08765738
# 3   1    2 2.197877e+01 615.0706 1166.7605 -0.08789426 -0.08764055 -0.08765680
# 4   1    3 2.974503e+00 435.9062 1217.1511 -0.08811989 -0.08765274 -0.08765646
# 5   1    4 4.025552e-01 339.0661 1207.3381 -0.08816317 -0.08766848 -0.08765659
# 6   1    5 5.447993e-02 287.5025 1170.4685 -0.08810975 -0.08768340 -0.08765720
# 7   1    6 7.373058e-03 257.6031 1122.8839 -0.08800312 -0.08769524 -0.08765823
# 8   1    7 9.978352e-04 237.8551 1072.0195 -0.08786379 -0.08770280 -0.08765953
# 9   1    8 1.350418e-04 222.9737 1021.1484 -0.08770143 -0.08770535 -0.08766094
# 10  1    9 1.827653e-05 210.5666  971.6636 -0.08752080 -0.08770241 -0.08766231
# 11  1   10 2.473336e-06 199.5489  924.1189 -0.08732440 -0.08769362 -0.08766345

e1 <- ev(amt = 1200, rate = 150, ii = 12, ss = 1, cmt = 1)
mod %>% ev(e1) %>% mrgsim(end = 500) %>% plot # plot data
time <- seq(from = 1, to = 10, by = 1)
xdata <- mod %>% ev(e1) %>% mrgsim(Req = "GUT, CENT, PERI, PROL, TRANSIT, CIRC",
                                   end = -1, add = time,
                                   rescort = 3) %>% as.data.frame
xdata
# ID time        GUT     CENT      PERI        PROL     TRANSIT        CIRC
# 1   1    0  0.0251597 232.5059 1022.9500 -0.08764767 -0.08769098 -0.08766079
# 2   1    1 64.8532585 281.8567  987.1464 -0.08751307 -0.08768729 -0.08766170
# 3   1    2 73.6267877 339.7994  981.0063 -0.08746266 -0.08768080 -0.08766240
# 4   1    3 74.8141559 373.5566  993.6797 -0.08747354 -0.08767398 -0.08766287
# 5   1    4 74.9748488 392.5755 1014.8106 -0.08751899 -0.08766832 -0.08766313
# 6   1    5 74.9965962 404.3396 1039.0644 -0.08758463 -0.08766462 -0.08766323
# 7   1    6 74.9995393 412.6384 1063.9958 -0.08766355 -0.08766332 -0.08766324
# 8   1    7 74.9999377 419.2312 1088.5357 -0.08775240 -0.08766471 -0.08766326
# 9   1    8 74.9999916 424.9186 1112.2394 -0.08784949 -0.08766899 -0.08766337
# 10  1    9 10.1501452 363.8037 1123.7893 -0.08791728 -0.08767598 -0.08766365
# 11  1   10  1.3736728 297.5622 1104.9980 -0.08788761 -0.08768335 -0.08766416
