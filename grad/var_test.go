package grad

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVar(t *testing.T) {
	assert := assert.New(t)

	v := NewVar(0.1)

	assert.Equal(0.1, v.value)
	assert.Equal(1.0, v.grad)
	assert.Equal("none", v.op)
	assert.Empty(v.children)

	assert.Equal("var(0.100000, op=<none>)", v.String())

	assert.Equal(0.1, v.Value())
	assert.Equal(1.0, v.Grad())
}
