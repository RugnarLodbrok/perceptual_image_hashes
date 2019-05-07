from os.path import join as j

from hashes import avghash, phash, pwhash

if __name__ == '__main__':
    for hash_function in [avghash, phash, pwhash]:
        d = {}
        for p in ['github_logo1.PNG', 'github_logo2.PNG', 'github_logo3.PNG']:
            ph = hash_function(j('..', 'data', p))
            d[p] = ph
            print(p, bin(ph))
        phashes = list(d.values())
        for i in range(len(phashes)):
            i2 = i + 1
            if i2 == len(phashes):
                i2 = 0
            diff = bin(phashes[i] ^ phashes[i2]).count('1')
            print(diff)
        # show(idct2d(ff))
