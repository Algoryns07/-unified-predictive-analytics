# MetroPT GPS Speed Quality Assessment

## Dataset scope

- High-speed threshold: >100 km/h
- High-speed observations: 75
- Observations with at least one valid coordinate comparison: 62
- Observations without a valid comparison: 13
- Maximum allowed adjacent time gap during coordinate analysis: 5 seconds

## Reported GPS speed

count     75.000
mean     143.067
std       47.315
min      102.000
25%      110.000
50%      127.000
75%      159.000
max      323.000

## Coordinate-derived speed

count     62.000
mean     179.097
std      130.050
min        0.000
25%      104.031
50%      158.965
75%      232.160
max      872.185

## Comparison

- Pearson correlation: -0.0451
- Within 10 km/h: 9
- Within 25 km/h: 20
- Within 50 km/h: 30

Agreement counts use only observations with at least one valid
coordinate comparison (n=62).

Number of valid comparisons per observation:
{0: 13, 1: 32, 2: 30}

Agreement counts are descriptive sensitivity checks, not validated
GPS accuracy thresholds. Coordinate-derived speed is not independent
ground truth.

## High-speed episodes

- Number of episodes: 38
- Longest episode: 9 observations
- Maximum reported speed: 323.00 km/h
- Maximum coordinate-derived speed: 872.19 km/h

## Preliminary interpretation

The high reported GPS speeds are not independently validated by
coordinate-derived estimates. Some coordinate-derived speeds are
also unusually high.

Possible explanations include GPS positional jumps, measurement
noise, sampling effects, or genuine movement. The current analysis
does not establish which explanation is correct.

GPS observations should not be automatically deleted based only
on a speed threshold.

## Recommended handling

1. Preserve the original GPS measurements.
2. Retain GPS quality indicators.
3. Flag suspicious speed and coordinate transitions.
4. Avoid treating coordinate-derived speed as ground truth.
5. Conduct additional route and positional consistency checks
   before using GPS features in predictive modelling.
