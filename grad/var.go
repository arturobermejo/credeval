package grad

import (
	"fmt"
	"math"
)

type Var struct {
	value    float64
	op       string
	grad     float64
	children map[*Var]float64
}

func NewVar(value float64) *Var {
	return &Var{
		value: value,
		op:    "none",
		grad:  math.NaN(),
	}
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

func (v *Var) SetGrad(g float64) {
	v.grad = g

	for ch, derivative := range v.children {
		ch.SetGrad(g * derivative)
	}
}

func (v *Var) ZeroGrad() {
	v.grad = 0.0
}

func Sum(x, y *Var) *Var {
	n := NewVar(x.value + y.value)
	n.op = "sum"

	n.children = map[*Var]float64{
		x: 1,
		y: 1,
	}

	return n
}

func Mul(x, y *Var) *Var {
	n := NewVar(x.value * y.value)
	n.op = "mul"

	n.children = map[*Var]float64{
		x: y.value,
		y: x.value,
	}

	return n
}

func Sigmoid(x *Var) *Var {
	value := 1.0 / (1.0 + math.Exp(-x.value))

	n := NewVar(value)
	n.op = "sigmoid"

	n.children = map[*Var]float64{
		x: value * (1 - value),
	}

	return n
}

func BinaryCrossEntropy(pred, target *Var) (*Var, error) {
	if target.value != 0.0 && target.value != 1.0 {
		return nil, fmt.Errorf("target should be 0.0 or 1.0")
	}

	epsilon := 1e-15

	predNorm := math.Max(epsilon, pred.value)
	predNorm = math.Min(1-epsilon, pred.value)

	loss := -target.value*math.Log(predNorm) - (1-target.value)*math.Log(1-predNorm)

	n := NewVar(loss)
	n.op = "bce"

	d := (-target.value/predNorm + (1-target.value)/(1-predNorm))
	n.children = map[*Var]float64{
		pred:   d,
		target: d,
	}

	return n, nil
}
