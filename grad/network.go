package grad

import "math/rand"

type Neuron struct {
	w       *Var
	b       *Var
	actFunc func(*Var) *Var
}

func NewNeuron(actFunc func(*Var) *Var) *Neuron {
	return &Neuron{
		w:       NewVar(rand.Float64()),
		b:       NewVar(rand.Float64()),
		actFunc: actFunc,
	}
}

func (neu *Neuron) Call(input *Var) *Var {
	inputw := Mul(input, neu.w)
	r := Sum(inputw, neu.b)

	return neu.actFunc(r)
}

type Network struct {
	neurons  []*Neuron
	lossFunc func(*Var, *Var) (*Var, error)
}

func NewNetwork() *Network {
	return &Network{}
}

func (net *Network) AddNeuron(neu *Neuron) {
	net.neurons = append(net.neurons, neu)
}

func (net *Network) Forward(x, y *Var) (*Var, error) {
	input := x

	for _, neu := range net.neurons {
		input = neu.Call(input)
	}

	return net.lossFunc(input, y)
}
