import pstats
from pstats import SortKey
p = pstats.Stats('time')
p.sort_stats(SortKey.TIME).print_stats(10)
