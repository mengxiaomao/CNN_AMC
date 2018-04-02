function [M]=moment1(y,cy,p,q)
M=mean(y.^(p-q).*cy.^q);
end