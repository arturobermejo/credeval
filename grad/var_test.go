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
	assert.Empty(v.children)

	assert.Equal("var(0.100000, op=<none>)", v.String())

	assert.Equal(0.1, v.Value())
	assert.True(math.IsNaN(v.Grad()))
}

func TestSum(t *testing.T) {
	assert := assert.New(t)

	a := NewVar(0.7)
	b := NewVar(0.3)

	c := Sum(a, b)
	c.backward(1.0)

	assert.Equal(1.0, c.Grad())
	assert.Equal(1.0, a.Grad())
	assert.Equal(1.0, b.Grad())
}
