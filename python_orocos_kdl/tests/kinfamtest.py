# Copyright  (C)  2007  Ruben Smits <ruben dot smits at mech dot kuleuven dot be>

# Version: 1.0
# Author: Ruben Smits <ruben dot smits at mech dot kuleuven dot be>
# Maintainer: Ruben Smits <ruben dot smits at mech dot kuleuven dot be>
# URL: http://www.orocos.org/kdl

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


import unittest
from PyKDL import *
from math import *
import random

class KinfamTestFunctions(unittest.TestCase):
    
    def setUp(self):
        seg1 = Segment("seg1", Joint(Joint.RotZ),
                                 Frame(Vector(0.0,0.0,0.0)))
        seg2 = Segment("seg2", Joint(Joint.RotX),
                                 Frame(Vector(0.0,0.0,0.9)))
        seg3 = Segment("seg3", Joint(Joint.None),
                                 Frame(Vector(-0.4,0.0,0.0)))
        seg4 = Segment("seg4", Joint(Joint.RotY),
                             Frame(Vector(0.0,0.0,1.2)))
        seg5 = Segment("seg5", Joint(Joint.None),
                                 Frame(Vector(0.4,0.0,0.0)))
        seg6 = Segment("seg6", Joint(Joint.TransZ),
                                 Frame(Vector(0.0,0.0,1.4)))
        seg7 = Segment("seg7", Joint(Joint.TransX),
                                 Frame(Vector(0.0,0.0,0.0)))
        seg8 = Segment("seg8", Joint(Joint.TransY),
                                 Frame(Vector(0.0,0.0,0.4)))
        seg9 = Segment("seg9", Joint(Joint.None),
                                 Frame(Vector(0.0,0.0,0.0)))

        self.segments = (seg1, seg2, seg3, seg4, seg5, seg6, seg7, seg8, seg9)

        self.chain = Chain()
        map(self.chain.addSegment, self.segments)

        self.jacsolver   = ChainJntToJacSolver(self.chain)
        self.fksolverpos = ChainFkSolverPos_recursive(self.chain)
        self.fksolvervel = ChainFkSolverVel_recursive(self.chain)
        self.iksolvervel = ChainIkSolverVel_pinv(self.chain)
        self.iksolverpos = ChainIkSolverPos_NR(self.chain,self.fksolverpos,self.iksolvervel)

    def testBuildTree(self):
        self.tree = Tree()
        prev_segment_name = "root"
        for i in self.segments:
            next_segment_name = i.getName()
            add_seg_ok = self.tree.addSegment(i, prev_segment_name)
            msg = "could not add segment {0} to segment {1}".format(prev_segment_name, next_segment_name)
            self.assertTrue(add_seg_ok, msg)
            prev_segment_name = next_segment_name

    def testFkPosAndJac(self):
        deltaq = 1E-4
        epsJ = 1E-4

        F1=Frame()
        F2=Frame()

        q=JntArray(self.chain.getNrOfJoints())
        jac=Jacobian(self.chain.getNrOfJoints())
        
        for i in range(q.rows()):
            q[i]=random.uniform(-3.14,3.14)

        self.jacsolver.JntToJac(q,jac)
        
        for i in range(q.rows()):
            oldqi=q[i];
            q[i]=oldqi+deltaq
            self.assert_(0==self.fksolverpos.JntToCart(q,F2))
            q[i]=oldqi-deltaq
            self.assert_(0==self.fksolverpos.JntToCart(q,F1))
            q[i]=oldqi
            Jcol1 = diff(F1,F2,2*deltaq)
            Jcol2 = Twist(Vector(jac[0,i],jac[1,i],jac[2,i]),
                          Vector(jac[3,i],jac[4,i],jac[5,i]))
            self.assertEqual(Jcol1,Jcol2);

    def testFkVelAndJac(self):
        deltaq = 1E-4
        epsJ   = 1E-4
    
        q=JntArray(self.chain.getNrOfJoints())
        qdot=JntArray(self.chain.getNrOfJoints())
        for i in range(q.rows()):
            q[i]=random.uniform(-3.14,3.14)
            qdot[i]=random.uniform(-3.14,3.14)

        qvel=JntArrayVel(q,qdot);
        jac=Jacobian(self.chain.getNrOfJoints())

        cart = FrameVel.Identity();
        t = Twist.Zero();

        self.jacsolver.JntToJac(qvel.q,jac)
        self.assert_(self.fksolvervel.JntToCart(qvel,cart)==0)
        MultiplyJacobian(jac,qvel.qdot,t)
        self.assertEqual(cart.deriv(),t)

    def testFkVelAndIkVel(self):
        epsJ = 1E-7

        q=JntArray(self.chain.getNrOfJoints())
        qdot=JntArray(self.chain.getNrOfJoints())
        for i in range(q.rows()):
            q[i]=random.uniform(-3.14,3.14)
            qdot[i]=random.uniform(-3.14,3.14)

        qvel=JntArrayVel(q,qdot)
        qdot_solved=JntArray(self.chain.getNrOfJoints())
        
        cart = FrameVel()
        
        self.assert_(0==self.fksolvervel.JntToCart(qvel,cart))
        self.assert_(0==self.iksolvervel.CartToJnt(qvel.q,cart.deriv(),qdot_solved))
        
        self.assertEqual(qvel.qdot,qdot_solved);
        

    def testFkPosAndIkPos(self):
        q=JntArray(self.chain.getNrOfJoints())
        for i in range(q.rows()):
            q[i]=random.uniform(-3.14,3.14)
        
        q_init=JntArray(self.chain.getNrOfJoints())
        for i in range(q_init.rows()):
            q_init[i]=q[i]+0.1*random.random()
            
        q_solved=JntArray(q.rows())

        F1=Frame.Identity()
        F2=Frame.Identity()
    
        self.assert_(0==self.fksolverpos.JntToCart(q,F1))
        self.assert_(0==self.iksolverpos.CartToJnt(q_init,F1,q_solved))
        self.assert_(0==self.fksolverpos.JntToCart(q_solved,F2))
        
        self.assertEqual(F1,F2)
        self.assertEqual(q,q_solved)
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(KinfamTestFunctions('testFkPosAndJac'))
    suite.addTest(KinfamTestFunctions('testFkVelAndJac'))
    suite.addTest(KinfamTestFunctions('testFkVelAndIkVel'))
    suite.addTest(KinfamTestFunctions('testFkPosAndIkPos'))
    # return suite

#suite = suite()
#unittest.TextTestRunner(verbosity=3).run(suite)
            
