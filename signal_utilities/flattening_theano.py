import theano, numpy
from theano import tensor as T

# have a tuple of 4d tensors, with these dimensions:
dims = [(2,3,4,5), (1,3,4,9), (3,3,3,3)]
def product(ds) :
    out = 1
    for d in ds :
        out *= d
    return out
products = map(product, dims)

xs_vals = [numpy.random.ranf(dim) for dim in dims]

# flattening:
xs = []
for x_val in xs_vals :
    xs.append(T.tensor4())
combined = T.concatenate([T.flatten(x) for x in xs])

# now inverse mapping:
inverse_input = T.vector()
inverse_output = []
accum = 0
for prod, dim in zip(products, dims) :
    inverse_output.append(
            T.reshape(inverse_input[accum:accum+prod], dim)
        )
    accum += prod
    

flatten = theano.function(xs, combined)
inverse = theano.function([inverse_input], inverse_output)

flattened =  flatten(*xs_vals)
print "Flattened to get vector of size", flattened.shape
unflattened =  inverse(flattened)
print "Reversed to get", len(unflattened), "tensors of dims:", [x.shape for x in unflattened]
for true_x, our_x in zip(xs_vals, unflattened) :
    print "error:", numpy.sum((true_x - our_x)**2)