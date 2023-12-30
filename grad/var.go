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
}

func NewVar(value float64) *Var {
	v := &Var{
		value: value,
		op:    "none",
		grad:  math.NaN(),
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

func (v *Var) SetValue(value float64) {
	v.value = value
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

func BinaryCrossEntropy(pred, target *Var) (*Var, error) {
	if target.value != 0.0 && target.value != 1.0 {
		return nil, fmt.Errorf("target should be 0.0 or 1.0")
	}

	epsilon := 1e-15

	predNorm := math.Max(epsilon, pred.value)
	predNorm = math.Min(1-epsilon, pred.value)

	loss := -target.value*math.Log(predNorm) - (1-target.value)*math.Log(1-predNorm)

	v := NewVar(loss)

	v.op = "bce"
	v.backward = func(g float64) {
		v.grad = g

		d := (-target.value/predNorm + (1-target.value)/(1-predNorm))

		pred.backward(d * g)
		target.backward(d * g)
	}

	return v, nil
}
