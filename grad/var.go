package grad

import (
	"fmt"
	"math"
)

type Var struct {
	value    float64
	op       string
	backward func()
	grad     float64
	children []*Var
}

func NewVar(v float64) *Var {
	return &Var{
		value:    v,
		op:       "none",
		backward: func() {},
		grad:     math.NaN(),
		children: make([]*Var, 0),
	}
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
