function ret = d_loss_by_d_model(model, data, wd_coefficient)
  % model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
  % model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
  % data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>
  % data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>

  % The returned object <ret> is supposed to be exactly like parameter <model>, i.e. it has fields ret.input_to_hid and ret.hid_to_class, and those are of the same shape as they are in <model>.
  % However, in <ret>, the contents of those matrices are gradients (d loss by d weight), instead of weights.
	 
  % This is the only function that you're expected to change. Right now, it just returns a lot of zeros, which is obviously not the correct output. Your job is to change that.
  
  [~,numcases] = size(data.inputs);
  [hid_input, hid_output, class_input, log_class_prob, class_prob] = forward_pass(model, data);
  ret.input_to_hid = 1/(numcases)*(model.hid_to_class'*(class_prob-data.targets)).*(hid_output-hid_output.^2)*data.inputs'+wd_coefficient*model.input_to_hid;
  ret.hid_to_class = 1/(numcases)*(class_prob-data.targets)*hid_output'+wd_coefficient*model.hid_to_class;
end 