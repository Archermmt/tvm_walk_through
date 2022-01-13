//for DependencyGraph, add to DependencyGraph class
void visualize(const std::string& file_path){
  std::unordered_map<Node*,std::string> node_names;
  int cnt=0;
  std::ofstream out(file_path,std::ofstream::binary);
  if (out.is_open()){
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());
    //build map
    std::unordered_map<Node*, Expr> node_to_expr;
    for (auto expr_node : this->expr_node) {
      node_to_expr[expr_node.second] = expr_node.first;
    }
    //write the nodes
    std::cout<<"name : \"dependency\"\n";
    for (auto it = this->post_dfs_order.rbegin(); it != this->post_dfs_order.rend(); ++it) {
      DependencyGraph::Node* n = *it;
      auto iit = n->parents.head;
      if(node_names.find(n)==node_names.end()){
        node_names[n]="Node_"+std::to_string(cnt++);
      }
      std::cout<<"layer {  name:\""<<node_names[n]<<"\"\n";
      //add topo information
      std::cout<<"  top : \""<<node_names[n]<<"\"\n";
      if(iit!=nullptr){
        for (; iit != nullptr; iit = iit->next) {
          std::cout<<"  bottom : \""<<node_names[iit->value]<<"\"\n";
        }
      }
      //add type
      Expr expr = node_to_expr[n];
      if(!expr.defined()){
        std::cout<<"  type : \"Connect\"\n";
      }else if (expr.as<CallNode>()){
        auto call=Downcast<Call>(expr);
        auto op=Downcast<Op>(call->op);
        std::cout<<"  type : \"Call_"<<op->name<<"\"\n";
      }else if(expr.as<FunctionNode>()){
        std::cout<<"  type : \"Function\"\n";
      }else if(expr.as<TupleGetItemNode>()){
        auto node=Downcast<TupleGetItem>(expr);
        std::cout<<"  type : \"TupleGetItemNode\"\n";
      }else if(expr.as<OpNode>()){
        auto node=Downcast<Op>(expr);
        std::cout<<"  type : \"Op_"<<node->name<<"\"\n";
      }else if(expr.as<VarNode>()){
        auto node=Downcast<Var>(expr);
        std::cout<<"  type : \"Var\""<<"\n";
      }else{
        std::cout<<"  type : \"UNKNOWN\""<<"\n";
      }
      //add attributes
      std::cout<<"  layer_param : {\n";
      std::cout<<"    addr : \""<<n<<"\"\n";
      if(expr.as<TupleGetItemNode>()){
        auto node=Downcast<TupleGetItem>(expr);
        std::cout<<"    index : "<<node->index<<"\n";
      }else if(expr.as<VarNode>()){
        auto node=Downcast<Var>(expr);
        std::cout<<"    name_hint : \""<<node->name_hint()<<"\"\n";
      }
      std::cout<<"  }\n}\n";
    }
    std::cout.rdbuf(coutbuf);
    out.close();
  }
}

//base utils
std::string get_pattern_kind(const OpPatternKind& kind){
  std::string kind_name="kOpaque";
  switch(kind){
    case kElemWise:
      kind_name="kElemWise";
      break;
    case kBroadcast:
      kind_name="kBroadcast";
      break;
    case kInjective:
      kind_name="kInjective";
      break;
    case kCommReduce:
      kind_name="kCommReduce";
      break;
    case kOutEWiseFusable:
      kind_name="kOutEWiseFusable";
      break;
    case kTuple:
      kind_name="kTuple";
      break;
    default:
      break;
  }
  return kind_name;
}

//for IndexedForwardGraph, add to IndexedForwardGraph class
void visualize(const std::string& file_path){
  std::ofstream out(file_path,std::ofstream::binary);
  if (out.is_open()){
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());
    //write the nodes
    std::cout<<"name : \"dependency\"\n";
    for (auto it = this->post_dfs_order.rbegin(); it != this->post_dfs_order.rend(); ++it) {
      Node* n = *it;
      auto iit = n->outputs.head;
      std::cout<<"layer {  name:\"Node_"<<n->index<<"\"\n";
      //add topo information
      std::cout<<"  top : \"Node_"<<n->index<<"\"\n";
      if(iit!=nullptr){
        for (; iit != nullptr; iit = iit->next) {
          std::cout<<"  bottom : \"Node_"<<iit->value.node->index<<"\"\n";
        }
      }
      //add type
      auto expr=GetRef<ObjectRef>(n->ref);
      auto pattern_name=get_pattern_kind(n->pattern);
      if(!expr.defined()){
        std::cout<<"  type : \"Connect["<<pattern_name<<"]\"\n";
      }else if(expr.as<CallNode>()){
        auto call=Downcast<Call>(expr);
        auto op=Downcast<Op>(call->op);
        std::cout<<"  type : \"Call_"<<op->name<<"["<<pattern_name<<"]\"\n";
      }else if(expr.as<ConstantNode>()){
        std::cout<<"  type : \"Constant["<<pattern_name<<"]\"\n";
      }else if(expr.as<FunctionNode>()){
        std::cout<<"  type : \"Function["<<pattern_name<<"]\"\n";
      }else if(expr.as<TupleGetItemNode>()){
        auto node=Downcast<TupleGetItem>(expr);
        std::cout<<"  type : \"TupleGetItemNode["<<pattern_name<<"]\"\n";
      }else if(expr.as<OpNode>()){
        auto node=Downcast<Op>(expr);
        std::cout<<"  type : \"Op_"<<node->name<<"["<<pattern_name<<"]\"\n";
      }else if(expr.as<VarNode>()){
        auto node=Downcast<Var>(expr);
        std::cout<<"  type : \"Var["<<pattern_name<<"]\""<<"\n";
      }else{
        std::cout<<"  type : \"UNKNOWN["<<pattern_name<<"]\""<<"\n";
      }
      //add attributes
      std::cout<<"  layer_param : {\n";
      std::cout<<"    extern_ref : \""<<(n->extern_ref ? "true" : "false")<<"\"\n";
      if(expr.as<TupleGetItemNode>()){
        auto node=Downcast<TupleGetItem>(expr);
        std::cout<<"    index : "<<node->index<<"\n";
      }else if(expr.as<ConstantNode>()){
        auto node=Downcast<Constant>(expr);
        std::cout<<"    tensor_type : \""<<node->tensor_type()<<"\"\n";
      }else if(expr.as<VarNode>()){
        auto node=Downcast<Var>(expr);
        std::cout<<"    name_hint : \""<<node->name_hint()<<"\"\n";
      }
      std::cout<<"  }\n}\n";
    }
    std::cout.rdbuf(coutbuf);
    out.close();
  }
}

//for DominatorTree, add to DominatorTree class
void visualize(const std::string& file_path){
  std::ofstream out(file_path,std::ofstream::binary);
  if (out.is_open()){
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());
    //write the nodes
    std::cout<<"name : \"dependency\"\n";
    for (auto it = this->nodes.rbegin(); it != this->nodes.rend(); ++it) {
      Node* node = *it;
      IndexedForwardGraph::Node* gnode=node->gnode;
      std::cout<<"layer {  name:\"Node_"<<gnode->index<<"\"\n";
      //add topo information
      std::cout<<"  top : \"Node_"<<gnode->index<<"\"\n";
      if(node->parent!=nullptr){
        std::cout<<"  bottom : \"Node_"<<node->parent->gnode->index<<"\"\n";
      }
      //add type
      auto expr=GetRef<ObjectRef>(gnode->ref);
      auto pattern_name=get_pattern_kind(node->pattern);
      if(!expr.defined()){
        std::cout<<"  type : \"Connect\n["<<pattern_name<<"]\"\n";
      }else if(expr.as<CallNode>()){
        auto call=Downcast<Call>(expr);
        auto op=Downcast<Op>(call->op);
        std::cout<<"  type : \"Call_"<<op->name<<"["<<pattern_name<<"]\"\n";
      }else if(expr.as<ConstantNode>()){
        std::cout<<"  type : \"Constant["<<pattern_name<<"]\"\n";
      }else if(expr.as<FunctionNode>()){
        std::cout<<"  type : \"Function["<<pattern_name<<"]\"\n";
      }else if(expr.as<TupleGetItemNode>()){
        std::cout<<"  type : \"TupleGetItemNode["<<pattern_name<<"]\"\n";
      }else if(expr.as<OpNode>()){
        auto e_node=Downcast<Op>(expr);
        std::cout<<"  type : \"Op_"<<e_node->name<<"["<<pattern_name<<"]\"\n";
      }else if(expr.as<VarNode>()){
        auto e_node=Downcast<Var>(expr);
        std::cout<<"  type : \"Var["<<pattern_name<<"]\""<<"\n";
      }else{
        std::cout<<"  type : \"UNKNOWN["<<pattern_name<<"]\""<<"\n";
      }
      //add attributes
      std::cout<<"  layer_param : {\n";
      std::cout<<"    depth : \""<<node->depth<<"\"\n";
      if(expr.as<TupleGetItemNode>()){
        auto e_node=Downcast<TupleGetItem>(expr);
        std::cout<<"    index : "<<e_node->index<<"\n";
      }else if(expr.as<ConstantNode>()){
        auto e_node=Downcast<Constant>(expr);
        std::cout<<"    tensor_type : \""<<e_node->tensor_type()<<"\"\n";
      }else if(expr.as<VarNode>()){
        auto e_node=Downcast<Var>(expr);
        std::cout<<"    name_hint : \""<<e_node->name_hint()<<"\"\n";
      }
      std::cout<<"  }\n}\n";
    }
    std::cout.rdbuf(coutbuf);
    out.close();
  }
}

//for GraphPartitioner, add to GraphPartitioner class
void visualize(const std::string& file_path){
  std::unordered_map<Group*,std::string> group_names;
  std::unordered_map<const tvm::Object*,std::string> ref_names;
  std::ofstream out(file_path,std::ofstream::binary);
  if (out.is_open()){
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());
    //build names map
    for (int i=0;i<groups_.size();i++) {
      Group* group = groups_[i];
      group_names[group]="Node_"+std::to_string(i);
      if(group->root_ref!=nullptr){
        ref_names[group->root_ref]="Node_"+std::to_string(i);
      }
    }
    //write the nodes
    std::cout<<"name : \"graph_paritioner\"\n";
    for (int i=0;i<groups_.size();i++) {
      Group* group = groups_[i];
      std::cout<<"layer {  name:\""<<group_names[group]<<"\"\n";
      //add topo information
      std::cout<<"  top : \""<<group_names[group]<<"\"\n";
      if(group->parent!=nullptr){
        std::cout<<"  bottom : \""<<group_names[group->parent]<<"\"\n";
      }
      //add type
      auto expr=GetRef<ObjectRef>(group->root_ref);
      auto pattern_name=get_pattern_kind(group->pattern);
      if(!expr.defined()){
        std::cout<<"  type : \"Connect\n["<<pattern_name<<"]\"\n";
      }else if(expr.as<CallNode>()){
        auto call=Downcast<Call>(expr);
        auto op=Downcast<Op>(call->op);
        std::cout<<"  type : \"Call_"<<op->name<<"["<<pattern_name<<"]\"\n";
      }else if(expr.as<ConstantNode>()){
        std::cout<<"  type : \"Constant["<<pattern_name<<"]\"\n";
      }else if(expr.as<FunctionNode>()){
        std::cout<<"  type : \"Function["<<pattern_name<<"]\"\n";
      }else if(expr.as<TupleGetItemNode>()){
        std::cout<<"  type : \"TupleGetItemNode["<<pattern_name<<"]\"\n";
      }else if(expr.as<OpNode>()){
        auto e_node=Downcast<Op>(expr);
        std::cout<<"  type : \"Op_"<<e_node->name<<"["<<pattern_name<<"]\"\n";
      }else if(expr.as<VarNode>()){
        auto e_node=Downcast<Var>(expr);
        std::cout<<"  type : \"Var["<<pattern_name<<"]\""<<"\n";
      }else{
        std::cout<<"  type : \"UNKNOWN["<<pattern_name<<"]\""<<"\n";
      }
      //add attributes
      std::cout<<"  layer_param : {\n";
      if(group->anchor_ref!=nullptr){
        std::cout<<"    anchor_ref : \""<<ref_names[group->anchor_ref]<<"\"\n";
      }
      if(expr.as<TupleGetItemNode>()){
        auto e_node=Downcast<TupleGetItem>(expr);
        std::cout<<"    index : "<<e_node->index<<"\n";
      }else if(expr.as<ConstantNode>()){
        auto e_node=Downcast<Constant>(expr);
        std::cout<<"    tensor_type : \""<<e_node->tensor_type()<<"\"\n";
      }else if(expr.as<VarNode>()){
        auto e_node=Downcast<Var>(expr);
        std::cout<<"    name_hint : \""<<e_node->name_hint()<<"\"\n";
      }
      std::cout<<"  }\n}\n";
    }
    std::cout.rdbuf(coutbuf);
    out.close();
  }
}
