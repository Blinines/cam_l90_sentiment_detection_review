# -*- coding: utf-8 -*-

def get_accuracy(y):
	count, tot = 0, 0
	for elt in y['NEG']:
		if elt == 0:
			count += 1
		tot += 1
	for elt in y['POS']:
		if elt == 1:
			count += 1
		tot += 1
	return float(count)/tot
    

def sign_test(y_1, y_2):
    # plus : clf_1 better than clf_2
    # minus : clf_2 better than clf_1
    # null : clf_1 and clf_2 predicted the same
    numbers = {'plus': 0, 'minus': 0, 'null': 0}

    for i in range(len(y_1['NEG'])):
        if y_1['NEG'][i] == 0 and y_2['NEG'][i] == 1:
            numbers['plus'] += 1
        elif y_1['NEG'][i] == 1 and y_2['NEG'][i] == 0:
            numbers['minus'] += 1
        else:
            numbers['null'] += 1
    
    for i in range(len(y_1['POS'])):
        if y_1['POS'][i] == 1 and y_2['NEG'][i] == 0:
            numbers['plus'] += 1
        elif y_1['POS'][i] == 0 and y_2['NEG'][i] == 1:
            numbers['minus'] += 1
        else:
            numbers['null'] += 1
    
    return numbers


def binomial(n, k):
    if 0 <= k <= n:
        num, denom = 1, 1
        for t in range(1, min(k, n - k) + 1):
            num *= n
            denom *= t
            n -= 1
        return num // denom
    else:
        return 0


def p_value(numbers, q):
    N = 2*int(numbers['null']/2) + numbers['plus'] + numbers['minus']
    k = int(numbers['null']/2) + min(numbers['plus'], numbers['minus'])

    res = 0
    for i in range(k+1):
        res += binomial(N, i) * (q**i) * ((1-q)**(N-i))
    return 2*res


# q = 0.5
# print(p_value({'null': 100, 'plus': 67, 'minus': 33}, q))
# print(p_value({'null': 113, 'plus': 50, 'minus': 37}, q))
# print(p_value({'null': 102, 'plus':66, 'minus': 32}, q))