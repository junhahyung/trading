import os
import glob


ret = {}
dirs = glob.glob('/home/nas1_temp/junhahyung/trading/output/*')
for d in dirs:
    test_name = d.split('/')[-1]
    t = os.path.join(d, 'summary.txt')
    try:
        with open(t, 'r') as fp:
            contents = fp.readlines()
            summary = contents[-3]
            ret[test_name] = summary
    except:
        pass


with open('/home/nas1_temp/junhahyung/trading/summary_all.txt', 'w') as fp:
    for key in ret.keys():
        fp.write(key+'\n')
        fp.write(ret[key]+'\n====\n')
