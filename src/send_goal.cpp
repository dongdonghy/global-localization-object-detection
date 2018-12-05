#include <ros/ros.h>  
#include <serial/serial.h>
#include <move_base_msgs/MoveBaseAction.h>  
#include <std_msgs/String.h>
#include <std_msgs/Empty.h>
#include <list>
#include <iostream> 
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <string>
#include <geometry_msgs/Twist.h>
#include <actionlib/client/simple_action_client.h>  
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
using namespace std;

int flagrfid = 0;
int flagfinger = 0;

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

void navigation_move(double x, double y, double theta)
{
  move_base_msgs::MoveBaseGoal goal;
  goal.target_pose.header.frame_id = "map";  
  goal.target_pose.header.stamp = ros::Time::now();
  goal.target_pose.pose.position.x = x;  
  goal.target_pose.pose.position.y = y;
  goal.target_pose.pose.orientation=tf::createQuaternionMsgFromYaw(theta);
  MoveBaseClient ac("move_base", true);     
  while(!ac.waitForServer(ros::Duration(5.0))){  
    ROS_INFO("Waiting for the move_base action server to come up");  
  }  
  ROS_INFO("Sending goal");  
  ac.sendGoal(goal);  
  
  //ac.waitForResult();  
  
  //if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)  
  //  ROS_INFO("move secceed");  
  //else  
  //  ROS_INFO("The base failed to move to goal for some reason");    
}



void voice_goal_Callback(const std_msgs::String::ConstPtr& msg)
{
  string lab_A302="a302";
  string lab_B308="b308";
  string water_tank="\u8336\u6C34\u95F4";
  string statue="\u96D5\u5851";
  string mail_box="\u90AE\u7BB1";
  string yuandi="\u539F\u5730";

  string::size_type idx;
  idx=msg->data.find(lab_A302);
  if(idx!=string::npos)
    {
      cout<< "going to a302" <<endl;
      navigation_move(32.883, -29.847, -0.029);
    }


  idx=msg->data.find(lab_B308);
  if(idx!=string::npos)
    {
      cout<< "going to b308" <<endl;
      navigation_move(27.624, 1.695, -1.607);
    }

  idx=msg->data.find(water_tank);
  if(idx!=string::npos)
    {
      cout<< "going to water tank" <<endl;
      navigation_move(14.561, 1.994, -1.549);
    }

  idx=msg->data.find(statue);
  if(idx!=string::npos)
    {
      cout<< "going to statue" <<endl;
      navigation_move(32.884, -21.675, -3.137);
    }

  idx=msg->data.find(mail_box);
  if(idx!=string::npos)
    {
      cout<< "going to mail box" <<endl;
      navigation_move(27.675, -18.637, -0.008);
    }

  idx=msg->data.find(yuandi);
  if(idx!=string::npos)
    {
      cout<< "going to original place" <<endl;
      navigation_move(31.33, -17.2, 1.57);
    }

}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "send_goal");  

  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe("/Rog_result", 1, voice_goal_Callback);

  ros::spin();
  return 0;
}






