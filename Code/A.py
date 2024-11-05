# How many legs 1 axex + 0 basilisks + 0 centaurs have?
print("1 0 0")
# Sphinx Answers
a = int(input())
# How many legs 0 axex + 1 basilisks + 0 centaurs have?
print("0 1 0")
# Sphinx Answers
b = int(input())
# How many legs 0 axex + 0 basilisks + 1 centaurs have?
print("0 0 1")
# Sphinx Answers
c = int(input())
# How many legs 1 axex + 1 basilisks + 1 centaurs have? We ask this to do a fact check to the last Sphinx answers
print("1 1 1")
# Sphinx Answers
d = int(input())
# How many legs 1 axex + 2 basilisks + 3 centaurs have? We ask this to do a secondary fact check to the last Sphinx answer
print("1 2 3")
# Sphinx Answers
e = int(input())

# If a + b + c == d then the Sphinx didn't lie and it's fine we print the answer
if a + b + c == d:
    print(a, b, c)
# If a + 2 * b + 3 * c == e then the Sphinx did lie in d but we tricked it to e we print the answer
elif a + 2 * b + 3 * c == e:
    print(a, b, c)
# We assume d is the total number but there is something wrong with a (axex legs) we try to find a through this logic
elif (d - b - c) + 2 * b + 3 * c == e:
    print(d - b - c, b, c)
# We assume d is the total number but there is something wrong with b (basilisk legs) we try to find a through this logic
elif a + 2 * (d - c - a) + 3 * c == e:
    print(a, d - c - a, c)
# We assume d is the total number but there is something wrong with c (centaur legs) we try to find a through this logic
else:
    print(a, b, d - a - b)