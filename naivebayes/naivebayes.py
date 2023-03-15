import numpy as np
import random


def setup():
    pass


def read_in_language_files(lang, min, max):
    words = []
    for i in range(min,max+1):
        filename = './languageID/'+lang+str(i)+'.txt'
        f = open(filename, 'r')
        lines = f.readlines()
        for line in lines:
            if not line.isspace():
                words.append(line.strip('\n'))
    return "".join([str(item) for item in words])

def char_to_idx(c):
    if c == ' ':
        return 0
    else:
        return ord(c) - 96

def get_lang_histogram(language_chars):
    language_histogram = [0] * 27
    for c in language_chars:
        if c == ' ':
            language_histogram[0] = language_histogram[0] + 1
        else:
            language_histogram[char_to_idx(c)] = language_histogram[char_to_idx(c)] + 1
    return language_histogram

def prior():
    return (10 + 0.5)/(30 + 3 * 0.5)

def print_latex_tabular(language_histogram):
    all_characters = ['space']+[chr(ord('a') + i) for i in range(0,26)]
    for char in all_characters:
        print(char, end=' & ')
    print()
    for count in language_histogram:
        print(count, end=' & ')
    print()
    print()

def get_class_conditional_probability_vector(language_histogram):
    total = np.sum(language_histogram)
    return [(i + 0.5)/ (total + 27*0.5) for i in language_histogram]

def calc_likelihood_of_x_given_lang(x, lang_probability_vector):
    product = 0
    for xi in x:
        # print(xi, end=' ')
        # print(char_to_idx(xi), end=' ')
        # print(lang_probability_vector[char_to_idx(xi)], end=' ')
        # print(np.log(lang_probability_vector[char_to_idx(xi)]))
        product = product + np.log(lang_probability_vector[char_to_idx(xi)])
    return product

################################################################################

def p1():
    e09 = read_in_language_files('e', 0, 9)
    s09 = read_in_language_files('s', 0, 9)
    j09 = read_in_language_files('j', 0, 9)

    print(['space']+[chr(ord('a') + i) for i in range(0,26)])
    print(get_lang_histogram(e09))
    print(get_lang_histogram(s09))
    print(get_lang_histogram(j09))

def p2():
    e09 = read_in_language_files('e', 0, 9)

    print(len(e09))
    print(np.sum(get_lang_histogram(e09)))

    print_latex_tabular(get_lang_histogram(e09))

    for p in get_class_conditional_probability_vector(get_lang_histogram(e09)):
        print(np.round(p, 4), end=', ')
    print()

def p3():
    s09 = read_in_language_files('s', 0, 9)
    j09 = read_in_language_files('j', 0, 9)

    print('SPANISH')
    for p in get_class_conditional_probability_vector(get_lang_histogram(s09)):
        print(np.round(p, 4), end=', ')
    print()
    print()

    print('JAPANESE')
    for p in get_class_conditional_probability_vector(get_lang_histogram(j09)):
        print(np.round(p, 4), end=', ')
    print()
    print()

def p4():
    e10 = read_in_language_files('e', 10, 10) # only e10.txt
    print(get_lang_histogram(e10))

def p5():
    e09 = read_in_language_files('e', 0, 9)
    s09 = read_in_language_files('s', 0, 9)
    j09 = read_in_language_files('j', 0, 9)

    x = read_in_language_files('e', 10, 10) # only e10.txt

    print('ENGLISH')
    for p in get_class_conditional_probability_vector(get_lang_histogram(e09)):
        print(np.round(p, 4), end=', ')
    print()
    print()

    print('SPANISH')
    for p in get_class_conditional_probability_vector(get_lang_histogram(s09)):
        print(np.round(p, 4), end=', ')
    print()
    print()

    print('JAPANESE')
    for p in get_class_conditional_probability_vector(get_lang_histogram(j09)):
        print(np.round(p, 4), end=', ')
    print()
    print()

    print('THE TEST DATA X')
    print(x)
    print()
    print()

    print('p(x|y=e)')
    p_x_given_e = calc_likelihood_of_x_given_lang(x, get_class_conditional_probability_vector(get_lang_histogram(e09)))
    print(p_x_given_e)
    print('p(x|y=s)')
    p_x_given_s = calc_likelihood_of_x_given_lang(x, get_class_conditional_probability_vector(get_lang_histogram(s09)))
    print(p_x_given_s)
    print('p(x|y=j)')
    p_x_given_j = calc_likelihood_of_x_given_lang(x, get_class_conditional_probability_vector(get_lang_histogram(j09)))
    print(p_x_given_j)

    language_likelihoods = [p_x_given_e, p_x_given_s, p_x_given_j]
    language_names = ['English', 'Spanish', 'Japanese']

    print('assuming all the same priors, x is most likely', language_names[language_likelihoods.index(max(language_likelihoods))])

def p6():
    pass

def p7():
    # Training set
    e09 = read_in_language_files('e', 0, 9)
    s09 = read_in_language_files('s', 0, 9)
    j09 = read_in_language_files('j', 0, 9)

    theta_e = get_class_conditional_probability_vector(get_lang_histogram(e09))
    theta_s = get_class_conditional_probability_vector(get_lang_histogram(s09))
    theta_j = get_class_conditional_probability_vector(get_lang_histogram(j09))

    # Testing set
    e1019 = [read_in_language_files('e', i, i) for i in range(10,20)]
    s1019 = [read_in_language_files('s', i, i) for i in range(10,20)]
    j1019 = [read_in_language_files('j', i, i) for i in range(10,20)]

    # Classify
    for y in [e1019, s1019, j1019]:
        for x in y:
            p_x_given_e = calc_likelihood_of_x_given_lang(x, theta_e)
            p_x_given_s = calc_likelihood_of_x_given_lang(x, theta_s)
            p_x_given_j = calc_likelihood_of_x_given_lang(x, theta_j)

            p_e_given_x = p_x_given_e + np.log(prior())
            p_s_given_x = p_x_given_s + np.log(prior())
            p_j_given_x = p_x_given_j + np.log(prior())

            language_posteriors = [p_e_given_x, p_s_given_x, p_j_given_x]
            language_names = ['English', 'Spanish', 'Japanese']

            print('x is most likely', language_names[language_posteriors.index(max(language_posteriors))])

def p8():
    # Training set
    e09 = read_in_language_files('e', 0, 9)
    s09 = read_in_language_files('s', 0, 9)
    j09 = read_in_language_files('j', 0, 9)

    theta_e = get_class_conditional_probability_vector(get_lang_histogram(e09))
    theta_s = get_class_conditional_probability_vector(get_lang_histogram(s09))
    theta_j = get_class_conditional_probability_vector(get_lang_histogram(j09))

    # Testing set
    e1019 = [read_in_language_files('e', i, i) for i in range(10,20)]
    s1019 = [read_in_language_files('s', i, i) for i in range(10,20)]
    j1019 = [read_in_language_files('j', i, i) for i in range(10,20)]

    # Classify
    for y in [e1019, s1019, j1019]:
        for x in y:
            # Shuffle the words to prove order doesn't matter
            words = x.split()
            random.shuffle(words)
            x = ' '.join(words)

            # Classify
            p_x_given_e = calc_likelihood_of_x_given_lang(x, theta_e)
            p_x_given_s = calc_likelihood_of_x_given_lang(x, theta_s)
            p_x_given_j = calc_likelihood_of_x_given_lang(x, theta_j)

            p_e_given_x = p_x_given_e + np.log(prior())
            p_s_given_x = p_x_given_s + np.log(prior())
            p_j_given_x = p_x_given_j + np.log(prior())

            language_posteriors = [p_e_given_x, p_s_given_x, p_j_given_x]
            language_names = ['English', 'Spanish', 'Japanese']

            print('x is most likely', language_names[language_posteriors.index(max(language_posteriors))])




#setup()
#p1()
#p2()
#p3()
#p4()
#p5()
#p6()
#p7()
p8()
