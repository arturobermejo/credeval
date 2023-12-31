package grad

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVar(t *testing.T) {
	assert := assert.New(t)

	v := NewVar(0.1)

	assert.Equal(0.1, v.value)
	assert.True(math.IsNaN(v.grad))
	assert.Equal("none", v.op)

	assert.Equal("var(0.100000, op=<none>)", v.String())

	assert.Equal(0.1, v.Value())
	assert.True(math.IsNaN(v.Grad()))
}

func TestVarSetValue(t *testing.T) {
	assert := assert.New(t)

	v := NewVar(0.1)

	v.SetValue(1.0)
	assert.Equal(1.0, v.Value())
}

func TestVarZeroGrad(t *testing.T) {
	assert := assert.New(t)

	v := NewVar(0.1)

	v.backward(5.0)
	v.ZeroGrad()

	assert.Equal(0.0, v.grad)
}

func TestSum(t *testing.T) {
	assert := assert.New(t)

	a := NewVar(0.7)
	b := NewVar(0.3)

	c := Sum(a, b)

	assert.Equal(1.0, c.value)
	assert.Equal("sum", c.op)

	c.backward(1.0)

	assert.Equal(1.0, c.grad)
	assert.Equal(1.0, a.grad)
	assert.Equal(1.0, b.grad)
}

func TestMul(t *testing.T) {
	assert := assert.New(t)

	a := NewVar(0.2)
	b := NewVar(0.3)

	c := Mul(a, b)

	assert.Equal(0.06, c.value)
	assert.Equal("mul", c.op)

	c.backward(1.0)

	assert.Equal(1.0, c.grad)
	assert.Equal(0.3, a.grad)
	assert.Equal(0.2, b.grad)
}

func TestSigmoid(t *testing.T) {
	assert := assert.New(t)

	a := NewVar(0.2)

	b := Sigmoid(a)

	assert.Equal(0.549833997312478, b.value)
	assert.Equal("sigmoid", b.op)

	b.backward(1.0)

	assert.Equal(1.0, b.grad)
	assert.Equal(0.24751657271185995, a.grad)
}

func TestBinaryCrossEntropy(t *testing.T) {
	assert := assert.New(t)

	a := NewVar(1.0)
	b := NewVar(1.0)

	c, error := BinaryCrossEntropy(a, b)

	assert.NoError(error)

	assert.Equal("bce", c.op)
	assert.Equal(9.992007221626415e-16, c.value)

	c.backward(1.0)

	assert.Equal(1.0, c.grad)
	assert.Equal(-1.000000000000001, a.grad)
	assert.Equal(-1.000000000000001, b.grad)
}

func TestBinaryCrossEntropyError(t *testing.T) {
	assert := assert.New(t)

	a := NewVar(1.0)
	b := NewVar(5.0)

	c, error := BinaryCrossEntropy(a, b)

	assert.Nil(c)
	assert.Error(error)
}
