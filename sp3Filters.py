import numpy

def sp3Filters():
    ''' Steerable pyramid filters.  Transform described  in:

        @INPROCEEDINGS{Simoncelli95b,
            TITLE = "The Steerable Pyramid: A Flexible Architecture for
                     Multi-Scale Derivative Computation",
            AUTHOR = "E P Simoncelli and W T Freeman",
            BOOKTITLE = "Second Int'l Conf on Image Processing",
            ADDRESS = "Washington, DC", MONTH = "October", YEAR = 1995 }

        Filter kernel design described in:

        @INPROCEEDINGS{Karasaridis96,
            TITLE = "A Filter Design Technique for 
                     Steerable Pyramid Image Transforms",
            AUTHOR = "A Karasaridis and E P Simoncelli",
            BOOKTITLE = "ICASSP", ADDRESS = "Atlanta, GA",
            MONTH = "May", YEAR = 1996 }  '''

    filters = {}
    filters['harmonics'] = numpy.array([1, 3])
    filters['mtx'] = (
        numpy.array([[0.5000, 0.3536, 0, -0.3536],
                  [-0.0000, 0.3536, 0.5000, 0.3536],
                  [0.5000, -0.3536, 0, 0.3536],
                  [-0.0000, 0.3536, -0.5000, 0.3536]]))
    filters['hi0filt'] = (
        numpy.array([[-4.0483998600E-4, -6.2596000498E-4, -3.7829999201E-5,
                    8.8387000142E-4, 1.5450799838E-3, 1.9235999789E-3,
                    2.0687500946E-3, 2.0898699295E-3, 2.0687500946E-3,
                    1.9235999789E-3, 1.5450799838E-3, 8.8387000142E-4,
                    -3.7829999201E-5, -6.2596000498E-4, -4.0483998600E-4],
                  [-6.2596000498E-4, -3.2734998967E-4, 7.7435001731E-4,
                    1.5874400269E-3, 2.1750701126E-3, 2.5626500137E-3,
                    2.2892199922E-3, 1.9755100366E-3, 2.2892199922E-3,
                    2.5626500137E-3, 2.1750701126E-3, 1.5874400269E-3,
                    7.7435001731E-4, -3.2734998967E-4, -6.2596000498E-4],
                  [-3.7829999201E-5, 7.7435001731E-4, 1.1793200392E-3,
                    1.4050999889E-3, 2.2253401112E-3, 2.1145299543E-3,
                    3.3578000148E-4, -8.3368999185E-4, 3.3578000148E-4,
                    2.1145299543E-3, 2.2253401112E-3, 1.4050999889E-3,
                    1.1793200392E-3, 7.7435001731E-4, -3.7829999201E-5],
                  [8.8387000142E-4, 1.5874400269E-3, 1.4050999889E-3,
                   1.2960999738E-3, -4.9274001503E-4, -3.1295299996E-3,
                   -4.5751798898E-3, -5.1014497876E-3, -4.5751798898E-3,
                   -3.1295299996E-3, -4.9274001503E-4, 1.2960999738E-3,
                   1.4050999889E-3, 1.5874400269E-3, 8.8387000142E-4],
                  [1.5450799838E-3, 2.1750701126E-3, 2.2253401112E-3,
                   -4.9274001503E-4, -6.3222697936E-3, -2.7556000277E-3,
                   5.3632198833E-3, 7.3032598011E-3, 5.3632198833E-3,
                   -2.7556000277E-3, -6.3222697936E-3, -4.9274001503E-4,
                   2.2253401112E-3, 2.1750701126E-3, 1.5450799838E-3],
                  [1.9235999789E-3, 2.5626500137E-3, 2.1145299543E-3,
                   -3.1295299996E-3, -2.7556000277E-3, 1.3962360099E-2,
                   7.8046298586E-3, -9.3812197447E-3, 7.8046298586E-3,
                   1.3962360099E-2, -2.7556000277E-3, -3.1295299996E-3,
                   2.1145299543E-3, 2.5626500137E-3, 1.9235999789E-3],
                  [2.0687500946E-3, 2.2892199922E-3, 3.3578000148E-4,
                   -4.5751798898E-3, 5.3632198833E-3, 7.8046298586E-3,
                   -7.9501636326E-2, -0.1554141641, -7.9501636326E-2,
                   7.8046298586E-3, 5.3632198833E-3, -4.5751798898E-3,
                   3.3578000148E-4, 2.2892199922E-3, 2.0687500946E-3],
                  [2.0898699295E-3, 1.9755100366E-3, -8.3368999185E-4,
                   -5.1014497876E-3, 7.3032598011E-3, -9.3812197447E-3,
                   -0.1554141641, 0.7303866148, -0.1554141641, 
                   -9.3812197447E-3, 7.3032598011E-3, -5.1014497876E-3,
                   -8.3368999185E-4, 1.9755100366E-3, 2.0898699295E-3],
                  [2.0687500946E-3, 2.2892199922E-3, 3.3578000148E-4,
                   -4.5751798898E-3, 5.3632198833E-3, 7.8046298586E-3,
                   -7.9501636326E-2, -0.1554141641, -7.9501636326E-2,
                   7.8046298586E-3, 5.3632198833E-3, -4.5751798898E-3,
                   3.3578000148E-4, 2.2892199922E-3, 2.0687500946E-3],
                  [1.9235999789E-3, 2.5626500137E-3, 2.1145299543E-3,
                   -3.1295299996E-3, -2.7556000277E-3, 1.3962360099E-2,
                   7.8046298586E-3, -9.3812197447E-3, 7.8046298586E-3,
                   1.3962360099E-2, -2.7556000277E-3, -3.1295299996E-3,
                   2.1145299543E-3, 2.5626500137E-3, 1.9235999789E-3],
                  [1.5450799838E-3, 2.1750701126E-3, 2.2253401112E-3,
                   -4.9274001503E-4, -6.3222697936E-3, -2.7556000277E-3,
                   5.3632198833E-3, 7.3032598011E-3, 5.3632198833E-3,
                   -2.7556000277E-3, -6.3222697936E-3, -4.9274001503E-4,
                   2.2253401112E-3, 2.1750701126E-3, 1.5450799838E-3],
                  [8.8387000142E-4, 1.5874400269E-3, 1.4050999889E-3,
                   1.2960999738E-3, -4.9274001503E-4, -3.1295299996E-3,
                   -4.5751798898E-3, -5.1014497876E-3, -4.5751798898E-3,
                   -3.1295299996E-3, -4.9274001503E-4, 1.2960999738E-3,
                   1.4050999889E-3, 1.5874400269E-3, 8.8387000142E-4],
                  [-3.7829999201E-5, 7.7435001731E-4, 1.1793200392E-3,
                    1.4050999889E-3, 2.2253401112E-3, 2.1145299543E-3,
                    3.3578000148E-4, -8.3368999185E-4, 3.3578000148E-4,
                    2.1145299543E-3, 2.2253401112E-3, 1.4050999889E-3,
                    1.1793200392E-3, 7.7435001731E-4, -3.7829999201E-5],
                  [-6.2596000498E-4, -3.2734998967E-4, 7.7435001731E-4,
                    1.5874400269E-3, 2.1750701126E-3, 2.5626500137E-3,
                    2.2892199922E-3, 1.9755100366E-3, 2.2892199922E-3,
                    2.5626500137E-3, 2.1750701126E-3, 1.5874400269E-3,
                    7.7435001731E-4, -3.2734998967E-4, -6.2596000498E-4],
                  [-4.0483998600E-4, -6.2596000498E-4, -3.7829999201E-5,
                    8.8387000142E-4, 1.5450799838E-3, 1.9235999789E-3,
                    2.0687500946E-3, 2.0898699295E-3, 2.0687500946E-3,
                    1.9235999789E-3, 1.5450799838E-3, 8.8387000142E-4,
                    -3.7829999201E-5, -6.2596000498E-4, -4.0483998600E-4]]))
    filters['lo0filt'] = (
        numpy.array([[-8.7009997515E-5, -1.3542800443E-3, -1.6012600390E-3,
                    -5.0337001448E-4, 2.5240099058E-3, -5.0337001448E-4,
                    -1.6012600390E-3, -1.3542800443E-3, -8.7009997515E-5],
                  [-1.3542800443E-3, 2.9215801042E-3, 7.5227199122E-3,
                    8.2244202495E-3, 1.1076199589E-3, 8.2244202495E-3,
                    7.5227199122E-3, 2.9215801042E-3, -1.3542800443E-3],
                  [-1.6012600390E-3, 7.5227199122E-3, -7.0612900890E-3,
                    -3.7694871426E-2, -3.2971370965E-2, -3.7694871426E-2,
                    -7.0612900890E-3, 7.5227199122E-3, -1.6012600390E-3],
                  [-5.0337001448E-4, 8.2244202495E-3, -3.7694871426E-2,
                    4.3813198805E-2, 0.1811603010, 4.3813198805E-2,
                    -3.7694871426E-2, 8.2244202495E-3, -5.0337001448E-4],
                  [2.5240099058E-3, 1.1076199589E-3, -3.2971370965E-2,
                   0.1811603010, 0.4376249909, 0.1811603010,
                   -3.2971370965E-2, 1.1076199589E-3, 2.5240099058E-3],
                  [-5.0337001448E-4, 8.2244202495E-3, -3.7694871426E-2,
                    4.3813198805E-2, 0.1811603010, 4.3813198805E-2,
                    -3.7694871426E-2, 8.2244202495E-3, -5.0337001448E-4],
                  [-1.6012600390E-3, 7.5227199122E-3, -7.0612900890E-3,
                    -3.7694871426E-2, -3.2971370965E-2, -3.7694871426E-2,
                    -7.0612900890E-3, 7.5227199122E-3, -1.6012600390E-3],
                  [-1.3542800443E-3, 2.9215801042E-3, 7.5227199122E-3,
                    8.2244202495E-3, 1.1076199589E-3, 8.2244202495E-3,
                    7.5227199122E-3, 2.9215801042E-3, -1.3542800443E-3],
                  [-8.7009997515E-5, -1.3542800443E-3, -1.6012600390E-3,
                    -5.0337001448E-4, 2.5240099058E-3, -5.0337001448E-4,
                    -1.6012600390E-3, -1.3542800443E-3, -8.7009997515E-5]]))
    filters['lofilt'] = (
        numpy.array([[-4.3500000174E-5, 1.2078000145E-4, -6.7714002216E-4,
                    -1.2434000382E-4, -8.0063997302E-4, -1.5970399836E-3,
                    -2.5168000138E-4, -4.2019999819E-4, 1.2619999470E-3,
                    -4.2019999819E-4, -2.5168000138E-4, -1.5970399836E-3,
                    -8.0063997302E-4, -1.2434000382E-4, -6.7714002216E-4,
                    1.2078000145E-4, -4.3500000174E-5],
                  [1.2078000145E-4, 4.4606000301E-4, -5.8146001538E-4,
                   5.6215998484E-4, -1.3688000035E-4, 2.3255399428E-3,
                   2.8898599558E-3, 4.2872801423E-3, 5.5893999524E-3,
                   4.2872801423E-3, 2.8898599558E-3, 2.3255399428E-3,
                   -1.3688000035E-4, 5.6215998484E-4, -5.8146001538E-4,
                   4.4606000301E-4, 1.2078000145E-4],
                  [-6.7714002216E-4, -5.8146001538E-4, 1.4607800404E-3,
                    2.1605400834E-3, 3.7613599561E-3, 3.0809799209E-3,
                    4.1121998802E-3, 2.2212199401E-3, 5.5381999118E-4,
                    2.2212199401E-3, 4.1121998802E-3, 3.0809799209E-3,
                    3.7613599561E-3, 2.1605400834E-3, 1.4607800404E-3,
                    -5.8146001538E-4, -6.7714002216E-4],
                  [-1.2434000382E-4, 5.6215998484E-4, 2.1605400834E-3,
                    3.1757799443E-3, 3.1846798956E-3, -1.7774800071E-3,
                    -7.4316998944E-3, -9.0569201857E-3, -9.6372198313E-3,
                    -9.0569201857E-3, -7.4316998944E-3, -1.7774800071E-3,
                    3.1846798956E-3, 3.1757799443E-3, 2.1605400834E-3,
                    5.6215998484E-4, -1.2434000382E-4],
                  [-8.0063997302E-4, -1.3688000035E-4, 3.7613599561E-3,
                    3.1846798956E-3, -3.5306399222E-3, -1.2604200281E-2,
                    -1.8847439438E-2, -1.7508180812E-2, -1.6485679895E-2,
                    -1.7508180812E-2, -1.8847439438E-2, -1.2604200281E-2,
                    -3.5306399222E-3, 3.1846798956E-3, 3.7613599561E-3,
                    -1.3688000035E-4, -8.0063997302E-4],
                  [-1.5970399836E-3, 2.3255399428E-3, 3.0809799209E-3,
                    -1.7774800071E-3, -1.2604200281E-2, -2.0229380578E-2,
                    -1.1091699824E-2, 3.9556599222E-3, 1.4385120012E-2,
                    3.9556599222E-3, -1.1091699824E-2, -2.0229380578E-2,
                    -1.2604200281E-2, -1.7774800071E-3, 3.0809799209E-3,
                    2.3255399428E-3, -1.5970399836E-3],
                  [-2.5168000138E-4, 2.8898599558E-3, 4.1121998802E-3,
                    -7.4316998944E-3, -1.8847439438E-2, -1.1091699824E-2,
                    2.1906599402E-2, 6.8065837026E-2, 9.0580143034E-2,
                    6.8065837026E-2, 2.1906599402E-2, -1.1091699824E-2,
                    -1.8847439438E-2, -7.4316998944E-3, 4.1121998802E-3,
                    2.8898599558E-3, -2.5168000138E-4],
                  [-4.2019999819E-4, 4.2872801423E-3, 2.2212199401E-3,
                    -9.0569201857E-3, -1.7508180812E-2, 3.9556599222E-3,
                    6.8065837026E-2, 0.1445499808, 0.1773651242,
                    0.1445499808, 6.8065837026E-2, 3.9556599222E-3,
                    -1.7508180812E-2, -9.0569201857E-3, 2.2212199401E-3,
                    4.2872801423E-3, -4.2019999819E-4],
                  [1.2619999470E-3, 5.5893999524E-3, 5.5381999118E-4,
                   -9.6372198313E-3, -1.6485679895E-2, 1.4385120012E-2,
                   9.0580143034E-2, 0.1773651242, 0.2120374441,
                   0.1773651242, 9.0580143034E-2, 1.4385120012E-2,
                   -1.6485679895E-2, -9.6372198313E-3, 5.5381999118E-4,
                   5.5893999524E-3, 1.2619999470E-3],
                  [-4.2019999819E-4, 4.2872801423E-3, 2.2212199401E-3,
                    -9.0569201857E-3, -1.7508180812E-2, 3.9556599222E-3,
                    6.8065837026E-2, 0.1445499808, 0.1773651242,
                    0.1445499808, 6.8065837026E-2, 3.9556599222E-3,
                    -1.7508180812E-2, -9.0569201857E-3, 2.2212199401E-3,
                    4.2872801423E-3, -4.2019999819E-4],
                  [-2.5168000138E-4, 2.8898599558E-3, 4.1121998802E-3,
                    -7.4316998944E-3, -1.8847439438E-2, -1.1091699824E-2,
                    2.1906599402E-2, 6.8065837026E-2, 9.0580143034E-2,
                    6.8065837026E-2, 2.1906599402E-2, -1.1091699824E-2,
                    -1.8847439438E-2, -7.4316998944E-3, 4.1121998802E-3,
                    2.8898599558E-3, -2.5168000138E-4],
                  [-1.5970399836E-3, 2.3255399428E-3, 3.0809799209E-3,
                    -1.7774800071E-3, -1.2604200281E-2, -2.0229380578E-2,
                    -1.1091699824E-2, 3.9556599222E-3, 1.4385120012E-2,
                    3.9556599222E-3, -1.1091699824E-2, -2.0229380578E-2,
                    -1.2604200281E-2, -1.7774800071E-3, 3.0809799209E-3,
                    2.3255399428E-3, -1.5970399836E-3],
                  [-8.0063997302E-4, -1.3688000035E-4, 3.7613599561E-3,
                    3.1846798956E-3, -3.5306399222E-3, -1.2604200281E-2,
                    -1.8847439438E-2, -1.7508180812E-2, -1.6485679895E-2,
                    -1.7508180812E-2, -1.8847439438E-2, -1.2604200281E-2,
                    -3.5306399222E-3, 3.1846798956E-3, 3.7613599561E-3,
                    -1.3688000035E-4, -8.0063997302E-4],
                  [-1.2434000382E-4, 5.6215998484E-4, 2.1605400834E-3,
                    3.1757799443E-3, 3.1846798956E-3, -1.7774800071E-3,
                    -7.4316998944E-3, -9.0569201857E-3, -9.6372198313E-3,
                    -9.0569201857E-3, -7.4316998944E-3, -1.7774800071E-3,
                    3.1846798956E-3, 3.1757799443E-3, 2.1605400834E-3,
                    5.6215998484E-4, -1.2434000382E-4],
                  [-6.7714002216E-4, -5.8146001538E-4, 1.4607800404E-3,
                    2.1605400834E-3, 3.7613599561E-3, 3.0809799209E-3,
                    4.1121998802E-3, 2.2212199401E-3, 5.5381999118E-4,
                    2.2212199401E-3, 4.1121998802E-3, 3.0809799209E-3,
                    3.7613599561E-3, 2.1605400834E-3, 1.4607800404E-3,
                    -5.8146001538E-4, -6.7714002216E-4],
                  [1.2078000145E-4, 4.4606000301E-4, -5.8146001538E-4,
                   5.6215998484E-4, -1.3688000035E-4, 2.3255399428E-3,
                   2.8898599558E-3, 4.2872801423E-3, 5.5893999524E-3,
                   4.2872801423E-3, 2.8898599558E-3, 2.3255399428E-3,
                   -1.3688000035E-4, 5.6215998484E-4, -5.8146001538E-4,
                   4.4606000301E-4, 1.2078000145E-4],
                  [-4.3500000174E-5, 1.2078000145E-4, -6.7714002216E-4,
                    -1.2434000382E-4, -8.0063997302E-4, -1.5970399836E-3,
                    -2.5168000138E-4, -4.2019999819E-4, 1.2619999470E-3,
                    -4.2019999819E-4, -2.5168000138E-4, -1.5970399836E-3,
                    -8.0063997302E-4, -1.2434000382E-4, -6.7714002216E-4,
                    1.2078000145E-4, -4.3500000174E-5]]))
    filters['bfilts'] = (
        numpy.array([[-8.1125000725E-4, 4.4451598078E-3, 1.2316980399E-2,
                    1.3955879956E-2,  1.4179450460E-2, 1.3955879956E-2,
                    1.2316980399E-2, 4.4451598078E-3, -8.1125000725E-4,
                    3.9103501476E-3, 4.4565401040E-3, -5.8724298142E-3,
                    -2.8760801069E-3, 8.5267601535E-3, -2.8760801069E-3,
                    -5.8724298142E-3, 4.4565401040E-3, 3.9103501476E-3,
                    1.3462699717E-3, -3.7740699481E-3, 8.2581602037E-3,
                    3.9442278445E-2, 5.3605638444E-2, 3.9442278445E-2,
                    8.2581602037E-3, -3.7740699481E-3, 1.3462699717E-3,
                    7.4700999539E-4, -3.6522001028E-4, -2.2522680461E-2,
                    -0.1105690673, -0.1768419296, -0.1105690673,
                    -2.2522680461E-2, -3.6522001028E-4, 7.4700999539E-4,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    0.0000000000, 0.0000000000, 0.0000000000,
                    -7.4700999539E-4, 3.6522001028E-4, 2.2522680461E-2,
                    0.1105690673, 0.1768419296, 0.1105690673,
                    2.2522680461E-2, 3.6522001028E-4, -7.4700999539E-4,
                    -1.3462699717E-3, 3.7740699481E-3, -8.2581602037E-3,
                    -3.9442278445E-2, -5.3605638444E-2, -3.9442278445E-2,
                    -8.2581602037E-3, 3.7740699481E-3, -1.3462699717E-3,
                    -3.9103501476E-3, -4.4565401040E-3, 5.8724298142E-3,
                    2.8760801069E-3, -8.5267601535E-3, 2.8760801069E-3,
                    5.8724298142E-3, -4.4565401040E-3, -3.9103501476E-3,
                    8.1125000725E-4, -4.4451598078E-3, -1.2316980399E-2,
                    -1.3955879956E-2, -1.4179450460E-2, -1.3955879956E-2,
                    -1.2316980399E-2, -4.4451598078E-3, 8.1125000725E-4],
                  [0.0000000000, -8.2846998703E-4, -5.7109999034E-5,
                   4.0110000555E-5, 4.6670897864E-3, 8.0871898681E-3,
                   1.4807609841E-2, 8.6204400286E-3, -3.1221499667E-3,
                   8.2846998703E-4, 0.0000000000, -9.7479997203E-4,
                   -6.9718998857E-3, -2.0865600090E-3, 2.3298799060E-3,
                   -4.4814897701E-3, 1.4917500317E-2, 8.6204400286E-3,
                   5.7109999034E-5, 9.7479997203E-4, 0.0000000000,
                   -1.2145539746E-2, -2.4427289143E-2, 5.0797060132E-2,
                   3.2785870135E-2, -4.4814897701E-3, 1.4807609841E-2,
                   -4.0110000555E-5, 6.9718998857E-3, 1.2145539746E-2,
                   0.0000000000, -0.1510555595, -8.2495503128E-2,
                   5.0797060132E-2, 2.3298799060E-3, 8.0871898681E-3,
                   -4.6670897864E-3, 2.0865600090E-3, 2.4427289143E-2,
                   0.1510555595, 0.0000000000, -0.1510555595,
                   -2.4427289143E-2, -2.0865600090E-3, 4.6670897864E-3,
                   -8.0871898681E-3, -2.3298799060E-3, -5.0797060132E-2,
                   8.2495503128E-2, 0.1510555595, 0.0000000000,
                   -1.2145539746E-2, -6.9718998857E-3, 4.0110000555E-5,
                   -1.4807609841E-2, 4.4814897701E-3, -3.2785870135E-2,
                   -5.0797060132E-2, 2.4427289143E-2, 1.2145539746E-2,
                   0.0000000000, -9.7479997203E-4, -5.7109999034E-5,
                   -8.6204400286E-3, -1.4917500317E-2, 4.4814897701E-3,
                   -2.3298799060E-3, 2.0865600090E-3, 6.9718998857E-3,
                   9.7479997203E-4, 0.0000000000, -8.2846998703E-4,
                   3.1221499667E-3, -8.6204400286E-3, -1.4807609841E-2,
                   -8.0871898681E-3, -4.6670897864E-3, -4.0110000555E-5,
                   5.7109999034E-5, 8.2846998703E-4, 0.0000000000],
                  [8.1125000725E-4, -3.9103501476E-3, -1.3462699717E-3,
                   -7.4700999539E-4, 0.0000000000, 7.4700999539E-4,
                   1.3462699717E-3, 3.9103501476E-3, -8.1125000725E-4,
                   -4.4451598078E-3, -4.4565401040E-3, 3.7740699481E-3,
                   3.6522001028E-4, 0.0000000000, -3.6522001028E-4,
                   -3.7740699481E-3, 4.4565401040E-3, 4.4451598078E-3,
                   -1.2316980399E-2, 5.8724298142E-3, -8.2581602037E-3,
                   2.2522680461E-2, 0.0000000000, -2.2522680461E-2,
                   8.2581602037E-3, -5.8724298142E-3, 1.2316980399E-2,
                   -1.3955879956E-2, 2.8760801069E-3, -3.9442278445E-2,
                   0.1105690673, 0.0000000000, -0.1105690673,
                   3.9442278445E-2, -2.8760801069E-3, 1.3955879956E-2,
                   -1.4179450460E-2, -8.5267601535E-3, -5.3605638444E-2,
                   0.1768419296, 0.0000000000, -0.1768419296,
                   5.3605638444E-2, 8.5267601535E-3, 1.4179450460E-2,
                   -1.3955879956E-2, 2.8760801069E-3, -3.9442278445E-2,
                   0.1105690673, 0.0000000000, -0.1105690673,
                   3.9442278445E-2, -2.8760801069E-3, 1.3955879956E-2,
                   -1.2316980399E-2, 5.8724298142E-3, -8.2581602037E-3,
                   2.2522680461E-2, 0.0000000000, -2.2522680461E-2,
                   8.2581602037E-3, -5.8724298142E-3, 1.2316980399E-2,
                   -4.4451598078E-3, -4.4565401040E-3, 3.7740699481E-3,
                   3.6522001028E-4, 0.0000000000, -3.6522001028E-4,
                   -3.7740699481E-3, 4.4565401040E-3, 4.4451598078E-3,
                   8.1125000725E-4, -3.9103501476E-3, -1.3462699717E-3,
                   -7.4700999539E-4, 0.0000000000, 7.4700999539E-4,
                   1.3462699717E-3, 3.9103501476E-3, -8.1125000725E-4],
                  [3.1221499667E-3, -8.6204400286E-3, -1.4807609841E-2,
                   -8.0871898681E-3, -4.6670897864E-3, -4.0110000555E-5,
                   5.7109999034E-5, 8.2846998703E-4, 0.0000000000,
                   -8.6204400286E-3, -1.4917500317E-2, 4.4814897701E-3,
                   -2.3298799060E-3, 2.0865600090E-3, 6.9718998857E-3,
                   9.7479997203E-4, -0.0000000000, -8.2846998703E-4,
                   -1.4807609841E-2, 4.4814897701E-3, -3.2785870135E-2,
                   -5.0797060132E-2, 2.4427289143E-2, 1.2145539746E-2,
                   0.0000000000, -9.7479997203E-4, -5.7109999034E-5,
                   -8.0871898681E-3, -2.3298799060E-3, -5.0797060132E-2,
                   8.2495503128E-2, 0.1510555595, -0.0000000000,
                   -1.2145539746E-2, -6.9718998857E-3, 4.0110000555E-5,
                   -4.6670897864E-3, 2.0865600090E-3, 2.4427289143E-2,
                   0.1510555595, 0.0000000000, -0.1510555595,
                   -2.4427289143E-2, -2.0865600090E-3, 4.6670897864E-3,
                   -4.0110000555E-5, 6.9718998857E-3, 1.2145539746E-2,
                   0.0000000000, -0.1510555595, -8.2495503128E-2,
                   5.0797060132E-2, 2.3298799060E-3, 8.0871898681E-3,
                   5.7109999034E-5, 9.7479997203E-4, -0.0000000000,
                   -1.2145539746E-2, -2.4427289143E-2, 5.0797060132E-2,
                   3.2785870135E-2, -4.4814897701E-3, 1.4807609841E-2,
                   8.2846998703E-4, -0.0000000000, -9.7479997203E-4,
                   -6.9718998857E-3, -2.0865600090E-3, 2.3298799060E-3,
                   -4.4814897701E-3, 1.4917500317E-2, 8.6204400286E-3,
                   0.0000000000, -8.2846998703E-4, -5.7109999034E-5,
                   4.0110000555E-5, 4.6670897864E-3, 8.0871898681E-3,
                   1.4807609841E-2, 8.6204400286E-3, -3.1221499667E-3]]).T)

    return filters
