function [final_vlogd, final_tlogd, valid_logDensity, train_logDensity, final_mixing] = mogem_3(numupdates, numgaussians, sdinit, mix_init, pausesecs);

%  mogem(numupdates, numgaussians, sdinit, pausesecs);
%Simple version of EM that fits a mixture of Gaussians to data.
%Assumes data has 2 columns.
%Assumes mixing proportions do not adapt and are all equal.

% numupdates: Number of EM iterations to run.
% numgaussians: Number of Gaussians to fit.
% sdinit: Initial standard deviation of the Gaussians.
% mix_init is the initial mixing proportions of the algo. 
% pausesecs: Number of seconds to pause after each EM iteration (so that the
%            visualization can be seen).


rand('seed', 2);
randn('seed',2);

if length(mix_init) ~= numgaussians
    fprintf('The initial mixing proportions does not match');
end

%% ASSUMES THAT train_data and valid_data HAVE BEEN CREATED ALREADY %%
global train_data;
global valid_data;
numcases   = size(train_data,1);
minvar     = .00001; %we get problems if variances get very small.

%% FIRST WE INITIALIZE THE PARAMETERS OF THE MODEL %%
centers    = rand(numgaussians,2);
xvars      = sdinit*sdinit * ones(1,numgaussians);
yvars      = sdinit*sdinit * ones(1,numgaussians);
%% THE MODEL IS NOW INITIALIZED %%

train_densities  = zeros(numcases, numgaussians);
valid_densities   = zeros(numcases, numgaussians);
%posteriors = train_densities;
train_xdata      = train_data(:,1)';
train_ydata      = train_data(:,2)';
%% INITIALIZE THE LOGDENSITIY RESULTS && MIXING PROPORTIONS
valid_logDensity = zeros(1, numupdates);
train_logDensity = zeros(1, numupdates);

mixing = mix_init;
%%

showmog; %% DISPLAYS OUR INITIAL MODEL AND THE DATA;
pause(pausesecs);

for i = 1:numupdates,

  % SET VALIDATION DENSITIES by adding densities contributed
  % by each Gaussian in the model. 
  % A 2-D density is the product of an x density and a y density.
  % The product of two 1-D Gaussians involves ADDING inside the exp.
  for g=1:numgaussians,
    xv= xvars(g);
    yv= yvars(g);
    xd = repmat(centers(g,1),numcases,1) - valid_data(:,1);
    yd = repmat(centers(g,2),numcases,1) - valid_data(:,2);
    valid_densities(:,g)= mixing(g)*(1/(2*pi*sqrt(xv*yv)))*...
                          exp(-xd.*xd/(2*xv) - yd.*yd/(2*yv));
    %%Scale by the mixing proportion
  end

  %SET VALIDATION POSTERIORS by normalizing the density under
  %each Gaussian by the summed density under all Gaussians.
  %sums = sum(valid_densities, 2)';
  %posteriors = valid_densities ./ repmat(sums',1,numgaussians);

%   fprintf(1, ' valid log prob = %4.5f    ', ...
%          sum(log(  sum(valid_densities,2)/numgaussians)));
    
  valid_logDensity(i)= sum(log(sum(valid_densities,2)/numgaussians));
  % Note that we are computing log of the probability *densities*. Therefore,
  % it is ok for the log to be positive.

  %% SET TRAIN DENSITIES %%
  for g=1:numgaussians,
    xv= xvars(g);
    yv= yvars(g);
    xd = repmat(centers(g,1),numcases,1) - train_data(:,1);
    yd = repmat(centers(g,2),numcases,1) - train_data(:,2);
    train_densities(:,g)= mixing(g)*(1/(2*pi*sqrt(xv*yv)))*...
                          exp(-xd.*xd/(2*xv) - yd.*yd/(2*yv));
    %%MODIFY LINE ABOVE IF USING UNEQUAL MIXING PROPORTIONS
  end

  %% SET TRAINING POSTERIORS %%
  sums = sum(train_densities, 2)';
  posteriors = train_densities ./ repmat(sums',1,numgaussians); % proportions fixed (E step)

  fprintf(1, ' train log prob = %4.5f \n ', ...
          sum(log(sum(train_densities,2)/numgaussians)));
  
  % Modify the way of computing log density if learning mixing proportions
  train_logDensity(i) = sum(log(sum(train_densities,2)/numgaussians));
  
  %% SET NEW CENTERS AND NEW VARIANCES %%
  for g=1:numgaussians,
    r        = posteriors(:,g);
    cx       = (train_xdata * r) / sum(r);
    cy       = (train_ydata * r) / sum(r);
    centers(g,:) = [cx cy];
    xnoise   = train_xdata - repmat(cx,1,numcases);
    xvars(g) = minvar + sum(r' .* xnoise .* xnoise)/ sum(r);
    ynoise   = train_ydata - repmat(cy,1,numcases);
    yvars(g) = minvar + sum(r' .* ynoise .* ynoise)/ sum(r);
    mixing(g) = sum(r)/length(r);
    %%TO UPDATE MIXING PROPORTIONS, ADD CODE HERE
  end

  showmog; %% DISPLAYS OUR CURRENT MODEL AND THE DATA;
  pause(pausesecs); %% SLOWS IT DOWN SO WE CAN SEE IT MOVE.
end
final_vlogd= sum(log(sum(valid_densities,2)/numgaussians));
final_tlogd = sum(log(sum(train_densities,2)/numgaussians));
final_mixing = mixing
