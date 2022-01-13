import tvm
from tvm import te
from tvm import relay
from tvm.relay.ty import TupleType,TensorType
from tvm.relay.expr_functor import ExprVisitor

def infer_type(node):
    """A method to infer the type of a relay expression."""
    mod = tvm.IRModule.from_expr(node)
    mod = relay.transform.InferType()(mod)
    return mod["main"].ret_type

class RelayVisualizer(ExprVisitor):
    def __init__(self,name="relay_ir"):
        super().__init__()
        self._name = name

    def reset(self):
        self._node_count = 0
        self._node_dict = {}
        self._node_names = []
        self._node_ids={}
        self._ignore_nodes=set()

    def visualize(self,mod,entry_name="main",path=None):
        #infer type before visit the entry function
        mod = relay.transform.InferType()(mod)
        self.reset()
        self.visit(mod[entry_name])
        #write graph to prototxt
        def _tensor_des(tensor):
            return '{{name:"{}",dtype:{},shape:{}}}'.format(
                tensor["name"],tensor["dtype"],list(tensor["shape"]))
        path=path or self._name+'.prototxt'
        with open(path,'w') as f:
            f.write('name : "{}"\n'.format(self._name))
            for k in self._node_names:
                if k in self._ignore_nodes:
                    continue
                node_des=self._node_dict[k]
                topo=['top:"{}"'.format(k)]+['bottom:"{}"'.format(p) for p in node_des.get('parents',[])]
                layer_param=['idx:'+str(node_des["idx"])]+ \
                    ['in_{} {}'.format(idx,_tensor_des(i)) for idx,i in enumerate(node_des.get("inputs",[]))]+ \
                    ['out_{} {}'.format(idx,_tensor_des(o)) for idx,o in enumerate(node_des.get("outputs",[]))]
                if "attrs" in node_des:
                    layer_param+=['attrs '+str(node_des["attrs"])]
                f.write('layer {{{0}name:"{1}"{0}type:"{2}"{0}{3}{0}layer_param {{{0}  {4}\n  }}\n}}\n'.format(
                    '\n  ',k,node_des["op"],'\n  '.join(topo),'\n    '.join(layer_param)))

    def visit(self,expr):
        super().visit(expr)
        self._node_count += 1

    def visit_var(self,var):
        node_des={
            "name"     : var.name_hint,
            "op"       : "input",
            "outputs"  : self._get_outputs(var,var.name_hint)
        }
        self._add_node(var,node_des)

    def visit_constant(self,const):
        node_des={
            "op"       : "const",
            "outputs"  : self._get_outputs(const)
        }
        self._add_node(const,node_des)

    def visit_tuple(self,tup):
        super().visit_tuple(tup)
        node_des={
            "op"       : "tuple",
        }
        self._update_depends(node_des,tup.fields)
        self._add_node(tup,node_des)

    def visit_function(self,func):
        for p in func.params:
            self.visit(p)
        self.visit(func.body)
        node_des={
            "op"       : "function",
            "outputs"  : self._get_outputs(func)
        }
        self._update_depends(node_des,list(func.params)+[func.body])
        self._add_node(func,node_des)

    def visit_let(self, let):
        super().visit_let(let)
        var_des=self._node_dict[self._node_ids[let.var]]
        var_des["op"]="var"
        value_des=self._node_dict[self._node_ids[let.value]]
        #update depends
        if "parents" not in var_des:
            var_des["parents"]=[]
            var_des["inputs"]=[]
        var_des["parents"].append(self._node_ids[let.value])
        var_des["inputs"].append(value_des["outputs"][0])
        node_des={
            "op"       : "let",
            "outputs"  : self._get_outputs(let)
        }
        self._update_depends(node_des,[let.var,let.body])
        self._add_node(let,node_des)

    def visit_call(self,call):
        if isinstance(call.op,tvm.ir.Op) and call.op.name=="memory.alloc_storage":
            node_des={
                "op"       : call.op.name.replace('.','_'),
                "outputs"  : [{
                    "name"     : "Node_{}:0".format(self._node_count),
                    "shape"    : [a.data.asnumpy().tolist() for a in call.args],
                    "dtype"    : str(call.attrs["dtype"])
                }],
                "attrs"    : {k:call.attrs[k] for k in call.attrs.keys()}
            }
            self._add_node(call,node_des)
            return
        if isinstance(call.op,tvm.ir.Op) and call.op.name=="memory.alloc_tensor":
            self.visit(call.args[0])
            node_des={
                "op"       : call.op.name.replace('.','_'),
                "outputs"  : [{
                    "name"     : "Node_{}:0".format(self._node_count),
                    "shape"    : call.args[2].data.asnumpy().tolist(),
                    "dtype"    : str(call.attrs["dtype"])
                }],
                "attrs"    : {k:call.attrs[k] for k in call.attrs.keys()}
            }
            self._update_depends(node_des,[call.args[0]])
            self._add_node(call,node_des)
            return
        super().visit_call(call)
        if isinstance(call.op,relay.Function):
            node_des={
                "op"       : "func_call",
                "outputs"  : self._get_outputs(call)
            }
            self._update_depends(node_des,list(call.args)+[call.op])
            self._add_node(call,node_des)
        elif isinstance(call.op,relay.Var):
            node_des={
                "op"       : "var_call",
                "outputs"  : self._get_outputs(call)
            }
            self._update_depends(node_des,[call.op]+list(call.args))
            self._add_node(call,node_des)
        elif isinstance(call.op,tvm.ir.GlobalVar):
            node_des={
                "name"     : call.op.name_hint,
                "op"       : "global_var",
                "outputs"  : [{"name":"{}:0".format(call.op.name_hint),"shape":[],"dtype":""}]
            }
            self._update_depends(node_des,call.args)
            self._add_node(call,node_des)
        elif isinstance(call.op,tvm.ir.Op) and call.op.name=="relay.op.annotation.simulated_quantize":
            node_des={
                "op"       : call.op.name.replace('.','_'),
                "outputs"  : self._get_outputs(call)
            }
            self._update_depends(node_des,call.args[:1])
            if call.attrs is not None and hasattr(call.attrs,"keys"):
                node_des["attrs"]={k:call.attrs[k] for k in call.attrs.keys()}
            for key,arg in zip(['scale','min','max'],call.args[1:]):
                node_name=self._node_ids[arg]
                node_des["attrs"][key]=self._node_dict[node_name]["outputs"][0]["name"]
                self._ignore_nodes.add(node_name)
            self._add_node(call,node_des)
        else:
            node_des={
                "op"       : call.op.name.replace('.','_'),
                "outputs"  : self._get_outputs(call)
            }
            self._update_depends(node_des,call.args)
            if call.attrs is not None and hasattr(call.attrs,"keys"):
                node_des["attrs"]={k:call.attrs[k] for k in call.attrs.keys()}
            self._add_node(call,node_des)

    def _update_depends(self,node_des,refs):
        parents,inputs=[],[]
        if len(refs)==1 and isinstance(refs[0],relay.Tuple):
            refs=refs[0]
        for r in refs:
            if isinstance(r,relay.TupleGetItem):
                p_name,out_idx=self._node_ids[r.tuple_value],int(r.index)
            else:
                p_name,out_idx=self._node_ids[r],0
            if p_name not in parents:
                parents.append(p_name)
            inputs.append(self._node_dict[p_name]["outputs"][out_idx])
        node_des.update({
            "parents" : parents,
            "inputs"  : inputs,
        })

    def _get_outputs(self,ref,node_name=None):
        node_name=node_name or "Node_"+str(self._node_count)
        if isinstance(ref,relay.Var):
            if isinstance(ref.type_annotation,tvm.ir.FuncType):
                output={
                    "name"   : "{}:0".format(node_name),
                    "shape"  : ref.type_annotation.ret_type.concrete_shape,
                    "dtype"  : ref.type_annotation.ret_type.dtype
                }
            elif isinstance(ref.type_annotation,(tvm.ir.TypeCall,tvm.ir.TupleType)):
                output={
                    "name"   : "{}:0".format(node_name),
                    "shape"  : [],
                    "dtype"  : ""
                }
            else:
                output={
                    "name"   : "{}:0".format(node_name),
                    "shape"  : ref.type_annotation.concrete_shape,
                    "dtype"  : ref.type_annotation.dtype
                }
            return [output]
        elif isinstance(ref,relay.Constant):
            output={
                "name"   : "{}:0".format(node_name),
                "shape"  : ref.checked_type.concrete_shape,
                "dtype"  : ref.checked_type.dtype
            }
            return [output]
        elif isinstance(ref,relay.Function):
            output={
                "name"   : "{}:0".format(node_name),
                "shape"  : ref.ret_type.concrete_shape,
                "dtype"  : ref.ret_type.dtype
            }
            return [output]
        elif isinstance(ref,relay.Let):
            return self._get_outputs(ref.var,node_name=node_name)
        elif isinstance(ref,relay.Call):
            infer_out = infer_type(ref)
            if isinstance(infer_out,TensorType):
                output={
                    "name"   : "{}:0".format(node_name),
                    "shape"  : infer_out.concrete_shape,
                    "dtype"  : infer_out.dtype
                }
                return [output]
            elif isinstance(infer_out,TupleType):
                outputs=[]
                for idx,t in enumerate(infer_out.fields):
                    output={
                        "name"   : "{}:{}".format(node_name,idx),
                        "shape"  : t.concrete_shape,
                        "dtype"  : t.dtype
                    }
                    outputs.append(output)
                return outputs
            else:
                raise Exception("Unexpected infer out for {}:{}".format(ref,infer_out))
        else:
            raise NotImplementedError("tensor reference {}({}) is not supported".format(ref,type(ref)))

    def _add_node(self,node_ref,node_des):
        if "name" in node_des:
            node_name=node_des.pop("name")
            if node_name in self._node_dict:
                cnt=1
                while node_name+'_'+str(cnt) in self._node_dict:
                    cnt+=1
                node_name+='_'+str(cnt)
        else:
            node_name="Node_"+str(self._node_count)
        node_des["idx"]=self._node_count
        if "outputs" not in node_des or not node_des["outputs"]:
            node_des["outputs"]=[{"name":"Node_{}:0".format(self._node_count),"shape":[],"dtype":""}]
        self._node_names.append(node_name)
        self._node_dict[node_name]=node_des
        self._node_ids[node_ref]=node_name
        #print("add node {}:{} with {}".format(node_name,node_des,node_ref))

def update_des(kwargs,obj,key=None):
    if hasattr(obj,key):
        des_list=str(getattr(obj,key)).split('\n')
    else:
        des_list=str(obj).split('\n')
    prefix=key or "des"
    if len(des_list)>1:
        prefix+='_'
    #simplify the codes
    des_list=[d.replace('{','').replace('}','').strip() for d in des_list]
    des_list=[d for d in des_list if d]
    space=0
    for idx,d in enumerate(des_list):
        des_list[idx]=' '*space+d
        if not d.startswith('//'):
            space+=1
    des_list=des_list[:10]
    if len(des_list)==1:
        kwargs[prefix]='"{}"'.format(des_list[0].replace('\"','\''))
    else:
        for i,d in enumerate(des_list):
            kwargs[prefix+'0'*(2-len(str(i)))+str(i)]='"{}"'.format(d.replace('\"','\''))

class PrimExprVisualizer(ExprVisitor):
    def __init__(self,name="prim_expr",simple_mode=True):
        super().__init__()
        self._name = name
        self._simple_mode = simple_mode

    def reset(self):
        self._node_count = 0
        self._node_dict = {}
        self._node_names = []
        self._node_ids={}
        self._tensor_producers={}

    def visualize(self,expr,path=None,name='main',simple_mode=None):
        #infer type before visit the entry function
        if simple_mode is not None:
            self._simple_mode=simple_mode
        self.reset()
        if isinstance(expr,tvm.IRModule):
            self.visit(expr[name])
        else:
            self.visit(expr)
        #write graph to prototxt
        path=path or self._name+'.prototxt'
        with open(path,'w') as f:
            f.write('name : "{}"\n'.format(self._name))
            for k in self._node_names:
                node_des=self._node_dict[k]
                topo=['top:"{}"'.format(k)]+['bottom:"{}"'.format(p) for p in node_des.get('parents',[])]
                layer_param=['idx:'+str(node_des["idx"])]
                if "attrs" in node_des:
                    layer_param+=['{}:{}'.format(k,v) for k,v in node_des["attrs"].items()]
                f.write('layer {{{0}name:"{1}"{0}type:"{2}"{0}{3}{0}layer_param {{{0}  {4}\n  }}\n}}\n'.format(
                    '\n  ',k,node_des["op"],'\n  '.join(topo),'\n    '.join(layer_param)))

    def visit(self,expr,ref_type=None):
        if expr in self._node_ids:
            return
        is_node=True
        if isinstance(expr,te.Tensor):
            self.visit_tensor(expr,ref_type)
            is_node=False
        elif isinstance(expr,te.PlaceholderOp):
            self.visit_placeholder(expr,ref_type)
        elif isinstance(expr,te.ComputeOp):
            self.visit_compute(expr,ref_type)
        elif isinstance(expr,tvm.ir.Array):
            self.visit_array(expr,ref_type)
        elif isinstance(expr,tvm.tir.Var):
            self.visit_var(expr,ref_type)
        elif isinstance(expr,tvm.tir.IterVar):
            self.visit_itervar(expr,ref_type)
        elif isinstance(expr,tvm.tir.IntImm):
            self.visit_imm(expr,"int",ref_type)
        elif isinstance(expr,tvm.tir.FloatImm):
            self.visit_imm(expr,"float",ref_type)
        elif isinstance(expr,tvm.tir.StringImm):
            self.visit_imm(expr,"string",ref_type)
        elif isinstance(expr,tvm.tir.Add):
            self.visit_elemwise(expr,"add",ref_type)
        elif isinstance(expr,tvm.tir.And):
            self.visit_elemwise(expr,"and",ref_type)
        elif isinstance(expr,tvm.tir.FloorDiv):
            self.visit_elemwise(expr,"floor_div",ref_type)
        elif isinstance(expr,tvm.tir.FloorMod):
            self.visit_elemwise(expr,"floor_mod",ref_type)
        elif isinstance(expr,tvm.tir.Max):
            self.visit_elemwise(expr,"max",ref_type)
        elif isinstance(expr,tvm.tir.Min):
            self.visit_elemwise(expr,"min",ref_type)
        elif isinstance(expr,tvm.tir.Mul):
            self.visit_elemwise(expr,"mul",ref_type)
        elif isinstance(expr,tvm.tir.GE):
            self.visit_elemwise(expr,"greater_equal",ref_type)
        elif isinstance(expr,tvm.tir.LE):
            self.visit_elemwise(expr,"less_equal",ref_type)
        elif isinstance(expr,tvm.tir.LT):
            self.visit_elemwise(expr,"less_than",ref_type)
        elif isinstance(expr,tvm.tir.Sub):
            self.visit_elemwise(expr,"sub",ref_type)
        elif isinstance(expr,tvm.tir.Allocate):
            self.visit_allocate(expr,ref_type)
        elif isinstance(expr,tvm.tir.AssertStmt):
            self.visit_assert(expr,ref_type)
        elif isinstance(expr,tvm.tir.AttrStmt):
            self.visit_attr(expr,ref_type)
        elif isinstance(expr,tvm.tir.Buffer):
            self.visit_buffer(expr,ref_type)
        elif isinstance(expr,tvm.tir.BufferLoad):
            self.visit_bufferload(expr,ref_type)
        elif isinstance(expr,tvm.tir.BufferRealize):
            self.visit_bufferrealize(expr,ref_type)
        elif isinstance(expr,tvm.tir.BufferStore):
            self.visit_bufferstore(expr,ref_type)
        elif isinstance(expr,tvm.tir.Call):
            self.visit_call(expr,ref_type)        
        elif isinstance(expr,tvm.tir.Evaluate):
            self.visit_evalute(expr,ref_type)        
        elif isinstance(expr,tvm.tir.For):
            self.visit_for(expr,ref_type)
        elif isinstance(expr,tvm.tir.IfThenElse):
            self.visit_ifthenelse(expr,ref_type)
        elif isinstance(expr,tvm.tir.LetStmt):
            self.visit_let(expr,ref_type)
        elif isinstance(expr,tvm.tir.Load):
            self.visit_load(expr,ref_type)
        elif isinstance(expr,tvm.tir.PrimFunc):
            self.visit_primfunc(expr,ref_type)
        elif isinstance(expr,tvm.tir.ProducerLoad):
            self.visit_producer(expr,ref_type)
        elif isinstance(expr,tvm.tir.Ramp):
            self.visit_ramp(expr,ref_type)
        elif isinstance(expr,tvm.tir.Reduce):
            self.visit_reduce(expr,ref_type)
        elif isinstance(expr,tvm.tir.SeqStmt):
            self.visit_seq(expr,ref_type)
        elif isinstance(expr,tvm.tir.Store):
            self.visit_store(expr,ref_type)
        elif isinstance(expr,tvm.tir.expr.CommReducer):
            self.visit_commreducer(expr,ref_type)
        elif isinstance(expr,tvm.ir.expr.Range):
            self.visit_range(expr,ref_type)
        else:
            super().visit(expr)
        if is_node:
            self._node_count += 1

    def visit_tensor(self,expr,ref_type=None):
        self._tensor_producers[expr]=expr.op
        self.visit(expr.op,ref_type)

    def visit_placeholder(self,expr,ref_type=None):
        node_des={
            "name"     : expr.name,
            "op"       : "place_holder",
            "attrs"    : {"dtype" : str(expr.dtype)}
        }
        self._add_node(expr,node_des,ref_type)

    def visit_compute(self,expr,ref_type=None):
        for b in expr.body:
            self.visit(b)
        node_des={
            "op"       : "compute",
        }
        node_des=self._append_parents(node_des,[b for b in expr.body])
        self._add_node(expr,node_des,ref_type)

    def visit_array(self,expr,ref_type=None):
        for idx,a in enumerate(expr):
            self.visit(a,"array_"+str(idx))
        node_des={
            "op"       : "array",
        }
        node_des=self._append_parents(node_des,[a for a in expr])
        self._add_node(expr,node_des,ref_type)

    def visit_var(self,expr,ref_type=None):
        node_des={
            "name"     : expr.name,
            "op"       : "var",
            "attrs"    : {"dtype":expr.dtype}
        }
        self._add_node(expr,node_des,ref_type)

    def visit_itervar(self,expr,ref_type=None):
        self.visit(expr.var,"iter")
        node_des={
            "op"       : "itervar",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"dom")
        update_des(node_des["attrs"],expr,"iter_type")
        update_des(node_des["attrs"],expr,"thread_tag")
        node_des=self._append_parents(node_des,[expr.var])
        self._add_node(expr,node_des,ref_type)

    def visit_imm(self,expr,optype,ref_type=None):
        node_des={
            "op"       : optype,
            "attrs"    : {"value":expr.value,"dtype":expr.dtype}
        }
        self._add_node(expr,node_des,ref_type)

    def visit_elemwise(self,expr,op_type,ref_type=None):
        self.visit(expr.a,"a")
        self.visit(expr.b,"b")
        node_des={
            "op"       : op_type,
        }
        node_des=self._append_parents(node_des,[expr.a,expr.b])
        self._add_node(expr,node_des,ref_type)

    def visit_allocate(self,expr,ref_type=None):
        if not self._simple_mode:
            self.visit(expr.buffer_var,"buffer")
        self.visit(expr.body)
        node_des={
            "op"       : "allocate",
            "attrs"    : {"dtype":str(expr.dtype)}
        }
        update_des(node_des["attrs"],expr,"extents")
        update_des(node_des["attrs"],expr,"condition")
        update_des(node_des["attrs"],expr,"body")
        if self._simple_mode:
            node_des=self._append_parents(node_des,[expr.body])
        else:
            node_des=self._append_parents(node_des,[expr.buffer_var,expr.body])
        self._add_node(expr,node_des,ref_type)
    
    def visit_assert(self,expr,ref_type=None):
        self.visit(expr.body)
        node_des={
            "op"       : "assert",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"condition")
        update_des(node_des["attrs"],expr,"message")
        node_des=self._append_parents(node_des,[expr.body])
        self._add_node(expr,node_des,ref_type)

    def visit_attr(self,expr,ref_type=None):
        parents=[]
        if not isinstance(expr.node,tvm.tir.StringImm):
            self.visit(expr.node,"node")
            parents.append(expr.node)
        self.visit(expr.body)
        node_des={
            "op"       : "attribute",
            "attrs"    : {"attr_key":expr.attr_key}
        }
        update_des(node_des["attrs"],expr,"body")
        update_des(node_des["attrs"],expr,"value")
        if isinstance(expr.node,tvm.tir.StringImm):
            update_des(node_des["attrs"],expr,"node")
        node_des=self._append_parents(node_des,parents+[expr.body])
        self._add_node(expr,node_des,ref_type)

    def visit_buffer(self,expr,ref_type=None):
        node_des={
            "op"       : "buffer",
            "attrs"    : {"shape":expr.shape,"dtype":expr.dtype}
        }
        update_des(node_des["attrs"],expr,"name")
        self._add_node(expr,node_des,ref_type)

    def visit_bufferload(self,expr,ref_type=None):
        self.visit(expr.buffer,"buffer")
        for i in expr.indices:
            self.visit(i,"indice")
        node_des={
            "op"       : "buffer_load",
        }
        node_des=self._append_parents(node_des,[expr.buffer]+list(expr.indices))
        self._add_node(expr,node_des,ref_type)

    def visit_bufferrealize(self,expr,ref_type=None):
        if not self._simple_mode:
            for idx,b in enumerate(expr.bounds):
                self.visit(b,"bound_"+str(idx))
        self.visit(expr.buffer,"buffer")
        self.visit(expr.body)
        node_des={
            "op"       : "buffer_realize",
            "attrs"    : {"condition":expr.condition}
        }
        update_des(node_des["attrs"],expr,"body")
        update_des(node_des["attrs"],expr,"bounds")
        if self._simple_mode:
            node_des=self._append_parents(node_des,[expr.body])
        else:
            node_des=self._append_parents(node_des,list(expr.bounds)+[expr.buffer,expr.body])
        self._add_node(expr,node_des,ref_type)

    def visit_bufferstore(self,expr,ref_type=None):
        self.visit(expr.buffer,"buffer")
        self.visit(expr.value,"value")
        if not self._simple_mode:
            for i in expr.indices:
                self.visit(i,"indice")
        node_des={
            "op"       : "buffer_store",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"value")
        update_des(node_des["attrs"],expr,"indices")
        if self._simple_mode:
            node_des=self._append_parents(node_des,[expr.buffer,expr.value])
        else:
            node_des=self._append_parents(node_des,[expr.buffer,expr.value]+list(expr.indices))
        self._add_node(expr,node_des,ref_type)

    def visit_call(self,expr,ref_type=None):
        record_args=not expr.op.name.startswith("tir.")
        if record_args:
            for a in expr.args:
                self.visit(a)
        node_des={
            "op"       : expr.op.name.replace('.','_'),
        }
        if record_args:
            node_des=self._append_parents(node_des,expr.args)
        else:
            node_des["attrs"]={}
            update_des(node_des["attrs"],expr,"body")
        self._add_node(expr,node_des,ref_type)

    def visit_commreducer(self,expr,ref_type=None):
        parents=[]
        for x in expr.lhs:
            self.visit(x,"reduce_l")
            parents.append(x)
        for x in expr.rhs:
            self.visit(x,"reduce_r")
            parents.append(x)
        for x in expr.result:
            self.visit(x,"reduce_res")
            parents.append(x)
        for x in expr.identity_element:
            self.visit(x,"reduce_ind")
            parents.append(x)
        node_des={
            "op"       : "common_reducer",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"result")
        node_des=self._append_parents(node_des,parents)
        self._add_node(expr,node_des,ref_type)

    def visit_evalute(self,expr,ref_type=None):
        self.visit(expr.value,"value")
        node_des={
            "op"       : "evaluate",
        }
        node_des=self._append_parents(node_des,[expr.value])
        self._add_node(expr,node_des,ref_type)

    def visit_for(self,expr,ref_type=None):
        if not self._simple_mode:
            self.visit(expr.loop_var,"loop_var")
            self.visit(expr.min,"for_min")
            self.visit(expr.extent,"for_extent")
        self.visit(expr.body)
        node_des={
            "op"       : "for",
            "attrs"    : {"kind":expr.kind}
        }
        update_des(node_des["attrs"],expr,"body")
        if self._simple_mode:
            node_des=self._append_parents(node_des,[expr.body])
        else:
            node_des=self._append_parents(node_des,[expr.loop_var,expr.min,expr.extent,expr.body])
        self._add_node(expr,node_des,ref_type)

    def visit_ifthenelse(self,expr,ref_type=None):
        parents=[]
        if expr.then_case:
            self.visit(expr.then_case,"true")
            parents.append(expr.then_case)
        if expr.else_case:
            self.visit(expr.else_case,"false")
            parents.append(expr.else_case)
        node_des={
            "op"       : "ifthenelse",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"condition")
        node_des=self._append_parents(node_des,parents)
        self._add_node(expr,node_des,ref_type)

    def visit_let(self,expr,ref_type=None):
        self.visit(expr.var,"var")
        self.visit(expr.value,"value")
        self.visit(expr.body)
        node_des={
            "op"       : "let",
        }
        node_des=self._append_parents(node_des,[expr.var,expr.value,expr.body])
        self._add_node(expr,node_des,ref_type)

    def visit_load(self,expr,ref_type=None):
        self.visit(expr.buffer_var,"load_buffer")
        self.visit(expr.index,"load_index")
        node_des={
            "op"       : "load",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"predicate")
        update_des(node_des["attrs"],expr,"body")
        node_des=self._append_parents(node_des,[expr.buffer_var,expr.index])
        self._add_node(expr,node_des,ref_type)

    def visit_primfunc(self,expr,ref_type=None):
        if not self._simple_mode:
            for k,v in expr.buffer_map.items():
                self.visit(v)
        self.visit(expr.body)
        node_des={
            "op"       : "primfunc",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"body")
        if self._simple_mode:
            node_des=self._append_parents(node_des,[expr.body])
        else:
            node_des=self._append_parents(node_des,[v for _,v in expr.buffer_map.items()]+[expr.body])
        self._add_node(expr,node_des,ref_type)

    def visit_producer(self,expr,ref_type=None):
        self.visit(expr.producer)
        for i in expr.indices:
            self.visit(i,"indice")
        node_des={
            "op"       : "producer_load",
        }
        node_des=self._append_parents(node_des,[expr.producer]+[i for i in expr.indices])
        self._add_node(expr,node_des,ref_type)

    def visit_ramp(self,expr,ref_type=None):
        self.visit(expr.base,"base")
        self.visit(expr.stride,"stride")
        node_des={
            "op"       : "ramp",
            "attrs"    : {"lanes":expr.lanes}
        }
        update_des(node_des["attrs"],expr,"base")
        update_des(node_des["attrs"],expr,"stride")
        node_des=self._append_parents(node_des,[expr.base,expr.stride])
        self._add_node(expr,node_des,ref_type)

    def visit_reduce(self,expr,ref_type=None):
        parents=[]
        self.visit(expr.combiner,"reduce_combiner")
        parents.append(expr.combiner)
        for s in expr.source:
            self.visit(s,"reduce_source")
            parents.append(s)
        for s in expr.init:
            self.visit(s,"reduce_init")
            parents.append(s)
        for s in expr.axis:
            self.visit(s,"reduce_axis")
            parents.append(s)
        node_des={
            "op"       : "reduce",
        }
        node_des=self._append_parents(node_des,parents)
        self._add_node(expr,node_des,ref_type)

    def visit_seq(self,expr,ref_type=None):
        for idx,e in enumerate(expr.seq):
            self.visit(e,"seq_"+str(idx))
        node_des={
            "op"       : "seq",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"seq")
        node_des=self._append_parents(node_des,expr.seq)
        self._add_node(expr,node_des,ref_type)

    def visit_store(self,expr,ref_type=None):
        self.visit(expr.buffer_var,"store_buffer")
        self.visit(expr.value,"store_value")
        self.visit(expr.index,"store_index")
        node_des={
            "op"       : "store",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"predicate")
        update_des(node_des["attrs"],expr,"value")
        update_des(node_des["attrs"],expr,"index")
        update_des(node_des["attrs"],expr,"body")
        node_des=self._append_parents(node_des,[expr.buffer_var,expr.value,expr.index])
        self._add_node(expr,node_des,ref_type)

    def visit_range(self,expr,ref_type=None):
        self.visit(expr.min)
        self.visit(expr.extent)
        node_des={
            "op"       : "range",
            "attrs"    : {}
        }
        update_des(node_des["attrs"],expr,"range")
        node_des=self._append_parents(node_des,[expr.min,expr.extent])
        self._add_node(expr,node_des,ref_type)

    def _get_node_des(self,expr):
        if isinstance(expr,te.Tensor):
            producer=self._tensor_producers[expr]
            return self._node_dict[self._node_ids[producer]]
        else:
            return self._node_dict[self._node_ids[expr]]

    def _append_parents(self,node_des,depends):
        node_des["parents"]=[]
        for d in depends:
            d_des=self._get_node_des(d)
            node_des["parents"].append(d_des["name"])
        return node_des

    def _add_node(self,node_ref,node_des,ref_type=None):
        if "name" not in node_des:
            try:
                if hasattr(node_ref,"name"):
                    node_des["name"]=getattr(node_ref,"name")
                else:
                    node_des["name"]="Node_"+str(self._node_count)
            except:
                node_des["name"]="Node_"+str(self._node_count)
        if ref_type:
            node_des["op"]="{}({})".format(node_des["op"],ref_type)
        #replica node name
        if node_des["name"] in self._node_dict:
            cnt=1
            while node_des["name"]+'_'+str(cnt) in self._node_dict:
                cnt+=1
            node_des["name"]=node_des["name"]+'_'+str(cnt)
        node_name=node_des["name"]
        node_des["idx"]=self._node_count
        self._node_names.append(node_name)
        self._node_dict[node_name]=node_des
        self._node_ids[node_ref]=node_name
        #print("add node {}:{} for {}".format(node_name,node_des,node_ref))
