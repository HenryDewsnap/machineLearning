##EXAMPLE:

m = 0.001
c = 0

def error(a,b): ##a=Desired, b=Actual
    out = []
    if len(a) != len(b): return out
    for i in range(len(a)): out.append(a[i]-b[i])
    return out

def func(x):
    return (x*m) + c

def get_adjustment(desired_data, actual_data):
    for i in range(len(actual_data)): actual_data[i] = func(actual_data[i])
    e = error(desired_data, actual_data)
    print(f"Error: {e}")

    return [
        e[0]*m ##Returning the error * the gradient as the adjustment.
        ]


def example():
    training_in = [
        [10,5,0],
        [0,5,10]
    ]
    training_ou = [
        [0],
        [1]
    ]

    weights = [0.5,0.5,0.5]
    for i in range(20000):
        output = weights[0] * training_in[i%len(training_in)][0] + weights[1] * training_in[i%len(training_in)][1] + weights[2] * training_in[i%len(training_in)][2]
        output = func(output)
        adjustments = get_adjustment(training_ou[i%len(training_ou)], [output])
        for wei in range(len(weights)):
            weights[wei] += adjustments[0] * training_in[i%2][wei]

    for i in range(2):
        print(func(training_in[i%len(training_in)][0] + weights[1] * training_in[i%len(training_in)][1] + weights[2] * training_in[i%len(training_in)][2]))

    print(func(0*weights[0] + 0*weights[1] + 0*weights[2]))

if __name__ == "__main__":
    example()
