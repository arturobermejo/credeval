package grad

import (
	"fmt"
	"math"
)

type Var struct {
	value    float64
	op       string
	backward func(float64)
	grad     float64
	children []*Var
}

func NewVar(value float64) *Var {
	v := &Var{
		value:    value,
		op:       "none",
		grad:     math.NaN(),
		children: make([]*Var, 0),
	}

	v.backward = func(g float64) {
		v.grad = g
	}

	return v
}

func (v *Var) String() string {
	return fmt.Sprintf("var(%f, op=<%s>)", v.value, v.op)
}

func (v *Var) Value() float64 {
	return v.value
}

func (v *Var) Grad() float64 {
	return v.grad
}

func Sum(a, b *Var) *Var {
	v := NewVar(a.value + b.value)

	v.op = "sum"
	v.backward = func(g float64) {
		v.grad = g

		a.backward(1.0 * g)
		b.backward(1.0 * g)
	}

	return v
}

func Mul(a, b *Var) *Var {
	v := NewVar(a.value * b.value)

	v.op = "mul"
	v.backward = func(g float64) {
		v.grad = g

		a.backward(b.value * g)
		b.backward(a.value * g)
	}

	return v
}

func Sigmoid(a *Var) *Var {
	value := 1.0 / (1.0 + math.Exp(-a.value))

	v := NewVar(value)

	v.op = "sigmoid"
	v.backward = func(g float64) {
		v.grad = g

		a.backward(value * (1 - value) * g)
	}

	return v
}
