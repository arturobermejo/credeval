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
